# hypernet_compressor_demo.py
# Ready-to-run PyTorch demo for a hypernetwork-based neural compressor (small image example).
# Produces a small synthetic image, trains a hypernetwork + per-patch latents,
# reconstructs the image, prints PSNR and size estimates, and shows images.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from math import log10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------- create a small structured image ----------------------
H_img, W_img = 256, 256
xs = np.linspace(0, 1, W_img)
ys = np.linspace(0, 1, H_img)
xx, yy = np.meshgrid(xs, ys)

img = np.zeros((H_img, W_img, 3), dtype=np.float32)
img[..., 0] = 0.5*(np.sin(10*xx) * np.cos(8.6*yy)) + 0.5*xx
img[..., 1] = 0.5*(np.sin(12*yy) * np.cos(6*np.pi*xx)) + 0.3*yy
g1 = np.exp(-((xx-0.3)**2 + (yy-0.6)**2) / (2*0.03))
g2 = np.exp(-((xx-0.7)**2 + (yy-0.4)**2) / (2*0.02))
img[..., 2] = 0.6*g1 + 0.5*g2 + 0.1*(1-xx)
img = np.clip(img, 0.0, 1.0)
orig_img = img.copy()

# ---------------------- patching ----------------------
patch_h, patch_w = 16, 16
assert H_img % patch_h == 0 and W_img % patch_w == 0
n_ph = H_img // patch_h
n_pw = W_img // patch_w
num_patches = n_ph * n_pw
print("num_patches:", num_patches)

# ---------------------- Fourier features for coordinates ----------------------
B = 8
ff_dim = 4 * B  # we generate sin/cos for x and y separately => 4*B dims

def fourier_features(coords, B=B):
    # coords: torch tensor shape [P,2] with coords in [0,1]
    freqs = (2.0 * np.pi * np.arange(B)).astype(np.float32)
    coords_np = coords.numpy()
    out = []
    for f in freqs:
        out.append(np.sin(f * coords_np[..., 0:1]))
        out.append(np.cos(f * coords_np[..., 0:1]))
        out.append(np.sin(f * coords_np[..., 1:2]))
        out.append(np.cos(f * coords_np[..., 1:2]))
    out = np.concatenate(out, axis=-1)
    return torch.from_numpy(out.astype(np.float32))

patches = []
for r in range(n_ph):
    for c in range(n_pw):
        y0, y1 = r*patch_h, (r+1)*patch_h
        x0, x1 = c*patch_w, (c+1)*patch_w
        patch = orig_img[y0:y1, x0:x1, :]  # [ph, pw, 3]
        xs_p = np.linspace(0, 1, patch_w)
        ys_p = np.linspace(0, 1, patch_h)
        xx_p, yy_p = np.meshgrid(xs_p, ys_p)
        coords = np.stack([xx_p, yy_p], axis=-1).astype(np.float32)
        coords_t = torch.from_numpy(coords.reshape(-1, 2))
        coords_ff = fourier_features(coords_t, B=B)
        targets = torch.from_numpy(patch.reshape(-1, 3).astype(np.float32))
        patches.append({"coords": coords_ff.to(device), "targets": targets.to(device)})

# ---------------------- decoder architecture & hypernetwork ----------------------
hidden = 64
out_dim = 3

def decoder_param_count(in_dim, h, out_dim):
    return in_dim*h + h + h*h + h + h*out_dim + out_dim

decoder_param_size = decoder_param_count(ff_dim, hidden, out_dim)
print("ff_dim:", ff_dim, "decoder_param_size:", decoder_param_size)

latent_dim = 16

class HyperNet(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, z):
        return self.net(z)

H_net = HyperNet(latent_dim, decoder_param_size).to(device)

# per-patch latent vectors (optimized directly)
Z = nn.ParameterList([nn.Parameter(torch.randn(latent_dim, device=device) * 0.01) for _ in range(num_patches)])

opt = torch.optim.Adam(list(H_net.parameters()) + list(Z), lr=5e-4)

# ---------------------- helper functions ----------------------
def parse_theta(theta_vec):
    ptr = 0
    n = ff_dim * hidden
    w1 = theta_vec[ptr:ptr+n].view(hidden, ff_dim); ptr += n
    n = hidden
    b1 = theta_vec[ptr:ptr+n].view(n); ptr += n
    n = hidden * hidden
    w2 = theta_vec[ptr:ptr+n].view(hidden, hidden); ptr += n
    n = hidden
    b2 = theta_vec[ptr:ptr+n].view(n); ptr += n
    n = hidden * out_dim
    w3 = theta_vec[ptr:ptr+n].view(out_dim, hidden); ptr += n
    n = out_dim
    b3 = theta_vec[ptr:ptr+n].view(n); ptr += n
    return w1, b1, w2, b2, w3, b3

def decoder_forward(coords_ff, theta_vec):
    w1, b1, w2, b2, w3, b3 = parse_theta(theta_vec)
    w1, b1, w2, b2, w3, b3 = [t.to(coords_ff.device) for t in (w1, b1, w2, b2, w3, b3)]
    x = F.linear(coords_ff, w1, b1)
    x = F.relu(x)
    x = F.linear(x, w2, b2)
    x = F.relu(x)
    x = F.linear(x, w3, b3)
    x = torch.sigmoid(x)
    return x

# ---------------------- training loop ----------------------
num_epochs = 5000
print_every = 50

for epoch in range(1, num_epochs+1):
    loss_sum = torch.tensor(0.0, device=device)
    # compute all thetas in a batch for speed
    z_batch = torch.stack([z for z in Z], dim=0)  # [num_patches, latent_dim]
    thetas = H_net(z_batch)                        # [num_patches, decoder_param_size]
    for i, p in enumerate(patches):
        coords = p["coords"]
        targets = p["targets"]
        theta = thetas[i]
        preds = decoder_forward(coords, theta)
        loss_sum = loss_sum + F.mse_loss(preds, targets) + 1e-4 * torch.mean(Z[i]**2)
    loss_sum.backward()
    opt.step()
    opt.zero_grad()
    if epoch % print_every == 0 or epoch == 1:
        print(f"Epoch {epoch}/{num_epochs}  total loss: {loss_sum.item():.6f}")

# ---------------------- reconstruction ----------------------
recon = np.zeros_like(orig_img)
with torch.no_grad():
    z_batch = torch.stack([z for z in Z], dim=0)
    thetas = H_net(z_batch)
    idx = 0
    for r in range(n_ph):
        for c in range(n_pw):
            preds = decoder_forward(patches[idx]["coords"], thetas[idx]).cpu().numpy()
            recon[r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w, :] = preds.reshape(patch_h, patch_w, 3)
            idx += 1
recon = np.clip(recon, 0.0, 1.0)

# ---------------------- metrics & rough compression estimate ----------------------
def psnr(a, b):
    mse = np.mean((a-b)**2)
    if mse == 0:
        return float('inf')
    return 10 * log10(1.0 / mse)

psnr_val = psnr(orig_img, recon)
print("PSNR:", psnr_val)

raw_bits = H_img * W_img * 3 * 8
H_params = sum(p.numel() for p in H_net.parameters())
H_bits_fp32 = H_params * 32
Z_count = num_patches * latent_dim
Z_bits_8 = Z_count * 8
total_bits_est = H_bits_fp32 + Z_bits_8
print(f"Raw image bits (8-bit RGB): {raw_bits:,}")
print(f"Hypernetwork params: {H_params} floats -> {H_bits_fp32:,} bits (fp32)")
print(f"Latents (quantized to 8-bit): {Z_bits_8:,} bits")
print("Estimated total bits (H fp32 + latents 8-bit):", f"{total_bits_est:,} bits")
# compression ratio ----------------------
compression_ratio = raw_bits / total_bits_est
print(f"Compression Ratio (raw / compressed): {compression_ratio:.2f}×")
# ---------------------- show images ----------------------
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(orig_img); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(recon); axes[1].set_title(f"Recon PSNR={psnr_val:.2f} dB"); axes[1].axis("off")
plt.show()

# End of hypernet_compressor_demo.py
