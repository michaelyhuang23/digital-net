import torch

B, C, H, W = 2, 3, 8, 6
x = torch.arange(B*C*H*W).view(B, C, H, W)

kernel_h, kernel_w = 2, 3
stride = 1

print(x.unfold(2, kernel_h, stride).shape)
patches = x.unfold(2, kernel_h, stride).unfold(3, kernel_w, stride)
print(patches.shape)