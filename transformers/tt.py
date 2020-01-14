import torch

indexes=torch.tensor([[0,1],[2,3],[8,9]],dtype=torch.long)

print(indexes)


flags=(indexes==2)

print(flags)

a=(not flags)

print(a)