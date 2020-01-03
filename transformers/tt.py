import torch

a= torch.nn.Parameter(torch.zeros([5,10]))
print(a)


nnb=torch.nn.Linear(5,6)

print(nnb.weight)

nnb.weight=torch.nn.Parameter(a[:,2:8])

print(nnb.weight)

a[:,3]=1

print(a)
print('********************************************')
print(nnb.weight)

a=torch.nn.Parameter(torch.zeros([5,10]))

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%')

print(a)
print('********************************************')
print(nnb.weight)

p1=torch.nn.Parameter(torch.rand(4,5))
p2=torch.nn.Parameter(torch.rand(2,5))

p3=torch.cat([p1,p2],dim=0)
print('********************************************')
print(p3)