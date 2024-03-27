import torch
from torch import nn
from einops import rearrange
import math

def get_conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1

class SignLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(SignLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.binarized_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.eps = 1e-6
        self.threshold = int(math.ceil(in_features / 2))
        self.root2 = math.sqrt(2)
        self.reset_parameters()

    def reset_parameters(self):  # initialization that preserves variance is non-trivial, we just use simple uniform here
        torch.nn.init.uniform_(self.weight, a=-2, b=2)

    def binarize_weight(self):
        self.binarized_weight.data = torch.bernoulli(torch.sigmoid(self.weight.data)) * 2 - 1
    
    def forward(self, p_x):
        if not self.training:
            return self.inference_forward(p_x)
        p_weight = torch.sigmoid(self.weight)
        # p_weight * p_x + (1-p_weight) * (1-p_x)
        # p_success = 2 * p_weight * p_x.unsqueeze(-2) + 1 - p_x.unsqueeze(-2) - p_weight  # (..., output_dim, input_dim)
        mu = (2 * p_x - 1) @ p_weight.t() + self.in_features - torch.sum(p_x, dim=-1, keepdim=True) # (..., output_dim)
        p_avg = mu / self.in_features
        sigma = torch.sqrt(p_avg * (1-p_avg) * self.in_features + self.eps) # (..., output_dim), std of binomial distribution

        p_actives = 0.5 - 0.5 * torch.erf((self.threshold - 0.5 - mu) / (sigma * self.root2)) # (..., output_dim)
        return p_actives
    
    def inference_forward(self, x):
        return torch.sign(x @ self.binarized_weight.t())
    


class SignConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(SignConv, self).__init__()
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels * kernel_size * kernel_size))
        self.binarized_weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels * kernel_size * kernel_size))
        self.in_features = in_channels * kernel_size * kernel_size
        self.out_features = out_channels
        self.eps = 1e-6
        self.threshold = int(math.ceil(self.in_features / 2))
        self.root2 = math.sqrt(2)
        self.reset_parameters()

    def reset_parameters(self):  # initialization that preserves variance is non-trivial, we just use simple uniform here
        torch.nn.init.uniform_(self.weight, a=-2, b=2)
    
    def binarize_weight(self):
        self.binarized_weight.data = torch.bernoulli(torch.sigmoid(self.weight.data)) * 2 - 1

    def forward(self, p_x): # (B, C, H, W)
        if not self.training:
            return self.inference_forward(p_x)
        batch_size = p_x.shape[0]
        output_height = get_conv_output_size(p_x.shape[-2], self.kernel_size, self.stride, self.padding)
        output_width = get_conv_output_size(p_x.shape[-1], self.kernel_size, self.stride, self.padding)
        expanded_p_x = self.unfold(p_x) # (B, C*kernel_size*kernel_size, H_out*W_out)
        expanded_p_x = rearrange(expanded_p_x, 'b c l -> (b l) c') 

        p_weight = torch.sigmoid(self.weight)
        # p_weight * p_x + (1-p_weight) * (1-p_x)
        # p_success = 2 * p_weight * p_x.unsqueeze(-2) + 1 - p_x.unsqueeze(-2) - p_weight  # (..., output_dim, input_dim)
        mu = (2 * expanded_p_x - 1) @ p_weight.t() + self.in_features - torch.sum(expanded_p_x, dim=-1, keepdim=True) # (..., output_dim)
        p_avg = mu / self.in_features
        sigma = torch.sqrt(p_avg * (1-p_avg) * self.in_features + self.eps) # (..., output_dim), std of binomial distribution

        p_actives = 0.5 - 0.5 * torch.erf((self.threshold - 0.5 - mu) / (sigma * self.root2)) # (..., output_dim)

        p_actives = rearrange(p_actives, '(b l) c -> b c l', b=batch_size) 
        folded_p_actives = p_actives.reshape(batch_size, self.out_features, output_height, output_width)

        return folded_p_actives

    def inference_forward(self, x):
        batch_size = x.shape[0]
        output_height = get_conv_output_size(x.shape[-2], self.kernel_size, self.stride, self.padding)
        output_width = get_conv_output_size(x.shape[-1], self.kernel_size, self.stride, self.padding)
        expanded_x = self.unfold(x) # (B, C*kernel_size*kernel_size, H_out*W_out)
        expanded_x = rearrange(expanded_x, 'b c l -> (b l) c') 

        expanded_x = torch.sign(expanded_x @ self.binarized_weight.t())

        expanded_x = rearrange(expanded_x, '(b l) c -> b c l', b=batch_size) 
        folded_x = expanded_x.reshape(batch_size, self.out_features, output_height, output_width)
        return folded_x


class SignOrPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(SignOrPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)

    def forward(self, p_x):
        if not self.training:
            return self.inference_forward(p_x)
        batch_size = p_x.shape[0]
        output_height = get_conv_output_size(p_x.shape[-2], self.kernel_size, self.stride, self.padding)
        output_width = get_conv_output_size(p_x.shape[-1], self.kernel_size, self.stride, self.padding)
        p_x = rearrange(p_x, 'b c h w -> (b c) 1 h w')
        expanded_p_x = self.unfold(p_x) # (B * C, 1*kernel_size*kernel_size, H_out*W_out)

        expanded_p_actives = 1 - (1 - expanded_p_x).prod(dim=-2, keepdim=True)
        folded_p_actives = expanded_p_actives.reshape(-1, 1, output_height, output_width)

        folded_p_actives = rearrange(folded_p_actives, '(b c) 1 h w -> b c h w', b=batch_size)
        return folded_p_actives

    def inference_forward(self, x):
        batch_size = x.shape[0]
        output_height = get_conv_output_size(x.shape[-2], self.kernel_size, self.stride, self.padding)
        output_width = get_conv_output_size(x.shape[-1], self.kernel_size, self.stride, self.padding)
        x = rearrange(x, 'b c h w -> (b c) 1 h w')
        expanded_x = self.unfold(x) # (B * C, 1*kernel_size*kernel_size, H_out*W_out)
        expanded_x = (expanded_x.sum(dim=-2, keepdim=True) > -(self.kernel_size * self.kernel_size - 0.5)).float() * 2 - 1
        folded_x = expanded_x.reshape(-1, 1, output_height, output_width)

        folded_x = rearrange(folded_x, '(b c) 1 h w -> b c h w', b=batch_size)
        return folded_x


class DigitalNet_1(nn.Module):
    def __init__(self):
        super(DigitalNet_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1)
        self.conv2 = SignConv(20, 50, kernel_size=3, stride=2, padding=0)
        self.conv3 = SignConv(50, 50, kernel_size=3, stride=2, padding=0)
        self.linear1 = SignLinear(50*5*5, 500)
        self.linear2 = SignLinear(500, 30)
        self.linear3 = nn.Linear(30, 10)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 50*5*5)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
    def binarize_weights(self):
        for module in self.children():
            if hasattr(module, 'binarize_weight'):
                module.binarize_weight()

##### TESTING ######
"""
if __name__ == '__main__':
    B, C, H, W = 2, 1, 30, 30
    x = torch.sigmoid(torch.arange(B*C*H*W).view(B, C, H, W).float())

    kernel_h, kernel_w = 2, 2
    stride = 2

    model = LeNet_5()

    print(model(x).shape)
"""