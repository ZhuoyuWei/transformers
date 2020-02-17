import sys
import os

input_dir=sys.argv[1]
ans_dir=r'/data/zhuoyu/semantic_parsing_v3/10data_3.0'

data_buffer=[]
for i in range(10):
    inputfile=os.path.join(ans_dir,str(i),'dev.ans')
    with open(inputfile,'r',encoding='utf-8') as f:
        for line in f:
            ss=line.strip().split('\t')
            if len(ss) == 3:
                data_buffer.append(ss)


ans_buffer=[]
for i in range(10):
    inputfile=os.path.join(input_dir,'{}_bert_output'.format(i),'dev.res')
    with open(inputfile,'r',encoding='utf-8') as f:
        for line in f:
            ss=line.strip()
            if len(ss) > 0:
                ans_buffer.append(ss)

print(len(data_buffer))
print(len(ans_buffer))
assert len(data_buffer) == len(ans_buffer)

with open(os.path.join(input_dir,'dev.total.res'),'w',encoding='utf-8') as fout:
    for i in range(len(ans_buffer)):
        data_buffer[i].append(ans_buffer[i])
        fout.write('\t'.join(data_buffer[i])+'\n')