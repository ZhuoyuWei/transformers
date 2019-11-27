import torch

b=torch.tensor([[1,2],[3,4]])
print(b.size())
b=b[:,None,None,:]
print(b.size())

a=torch.arange(10)

print(a)