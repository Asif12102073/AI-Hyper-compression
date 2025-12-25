# AI-Hyper-compression
A unique array of Neural networks to tackle data compression(with loss) for ordered data with minimal entropy. Currently, this algorithm can compress any image of functions (2 variables) or any ordered image and achieve a high compression ratio due to only storing the model weights. It uses a hypernetwork of NN (multiple models on different patches)

rev 1.0 image 
<img width="1722" height="899" alt="IMG comp" src="https://github.com/user-attachments/assets/f43f24af-4bfa-415c-a596-39d8c5113858" />

This early version has patching artifacts and is not dynamic. 
To fix this, a dynamic encoder needs to be implemented. Or any classifier model working in tandem to randomize the data.
