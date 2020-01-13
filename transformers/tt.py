import torch

inputs=torch.rand([3,10,4])
indexes=torch.tensor([[0,1],[2,3],[8,9]],dtype=torch.long)
print(inputs)
print(indexes)

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)



res=batched_index_select(inputs,1,indexes)
print(res)

print('################################################')
c=torch.tensor([[1,2],[3,4],[5,6]])

c=c-3
print(c)


d=(c<0)
print('################################################')
print(d)


print('################################################')
c=c.masked_fill(d,0)
print(c)

d=(c>2)
print('################################################')
print(d)


print('################################################')
e=c.masked_fill(d,0)
print(e)


a=torch.tensor([1,2,3])
b=torch.tensor([0,2,1])

c=(a>b)
print(c)
print('?????????????????????')
m=torch.rand([2,3,4])
n=torch.tensor([[0,1,0],[1,0,1]],dtype=torch.float).unsqueeze(-1).repeat(1,1,4)
print(m.size())
print(n.size())
print(n)
z=m*n
print(z)