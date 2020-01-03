import torch

a=torch.rand(5,10)
print(a)

b=torch.tensor([[0,1],[2,3]])

c=a.index_select(0,b.view([-1]))

print('b size={}'.format(b.size()))

c=c.view(list(b.size())+[-1])

print(c.size())
print(c)