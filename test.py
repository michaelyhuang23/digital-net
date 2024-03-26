import torch
import math

class SignLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(SignLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.eps = 1e-6
        self.threshold = int(math.ceil(in_features / 2))
        self.root2 = math.sqrt(2)
        self.reset_parameters()

    def reset_parameters(self):  # initialization that preserves variance is non-trivial, we just use simple uniform here
        torch.nn.init.uniform_(self.weight, a=-2, b=2)
    
    def forward(self, p_x):
        p_weight = torch.sigmoid(self.weight)
        # p_weight * p_x + (1-p_weight) * (1-p_x)
        p_success = 2 * p_weight * p_x.unsqueeze(-2) + 1 - p_x.unsqueeze(-2) - p_weight  # (..., output_dim, input_dim)

        mu = torch.sum(p_success, dim=-1) # (..., output_dim)
        sigma = torch.sqrt(torch.sum(p_success * (1 - p_success), dim=-1) + self.eps) # (..., output_dim), the mean of stds of each bernoulli 

        p_actives = 0.5 - 0.5 * torch.erf((self.threshold - 0.5 - mu) / (sigma * self.root2)) # (..., output_dim)
        return p_actives


# Usage
model = SignLinear(10, 20)
x = torch.rand(32, 10) # random in [0, 1]
y = model(x)

print(y.shape)  # torch.Size([32, 20])
print(y)

y.sum().backward()


print(model.weight.grad)

