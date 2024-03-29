import torch
from models.model import DigitalNet_1


model = DigitalNet_1()
pretrained_model = torch.load('models/LeNet_5.best.pth.tar')
model.load_state_dict(pretrained_model['state_dict'])

x = torch.rand(1, 1, 28, 28)


prob = model(x)
print(f'prob: {prob}') 
print(f'ev: {prob * 2 - 1}') 


model.eval()
outputs = []
with torch.no_grad():
    for _ in range(3000):
        model.binarize_weight()
        # ix = torch.bernoulli(x) * 2 - 1
        output = model(x)
        outputs.append(output)

print(torch.stack(outputs).mean(dim=0))
