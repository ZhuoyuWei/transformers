import sys

with open(sys.argv[1]) as f:
    scores=[0]*2
    for line in f:
        ss=line.strip().split('\t')
        if len(ss) == 2:
            for i in range(2):
                scores[i]+=float(ss[i])
    scores[0]/=10
    scores[1]/=10

    print('{}\t{}'.format(*scores))
