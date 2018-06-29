import numpy as np
import torch
b = torch.Tensor([[3,4],[7,8]])
a = torch.Tensor([[1,2,3,4],[4,5,6,7]])

for i in range(a.size()[1]):
    if a[:,i] in b[:,i]:
        a[:,i] = 0.0

print(a)