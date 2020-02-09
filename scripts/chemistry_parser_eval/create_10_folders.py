import sys
import codecs
import random
import os

data=[]
with codecs.open(sys.argv[1],'r','utf-8') as f:
    for line in f:
        ss=line.strip().split('\t')
        if len(ss)==3:
            data.append(ss)


outputdir=sys.argv[2]

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
else:
    print('ERROR: output dir exists')
    exit(-1)


random.seed(2020)
random.shuffle(data)
dev_size=len(data)//10

subdata=[]
for i in range(9):
    subdata.append(data[i*dev_size:(i+1)*dev_size])
subdata.append(data[9*dev_size:])
for i in range(10):
    train_buffer=[]
    for j in range(10):
        if i==j:
            continue
        train_buffer+=subdata[j]

    sub_dir=os.path.join(outputdir,str(i))
    os.makedirs(sub_dir)
    train_file=os.path.join(sub_dir,'train.tsv')
    dev_file=os.path.join(sub_dir,'dev.tsv')
    dev_ans_file=os.path.join(sub_dir,'dev.ans')

    with codecs.open(train_file,'w','utf-8') as fout:
        for ss in train_buffer:
            fout.write('\t'.join(ss)+'\n')

    with codecs.open(dev_file,'w','utf-8') as fout:
        for ss in subdata[i]:
            fout.write('\t'.join(ss[:-1])+'\n')

    with codecs.open(dev_ans_file,'w','utf-8') as fout:
        for ss in subdata[i]:
            fout.write('\t'.join(ss)+'\n')







