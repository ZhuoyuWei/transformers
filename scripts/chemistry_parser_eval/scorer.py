import sys
import codecs


answers=[]
with codecs.open(sys.argv[1],'r','utf-8') as f:
    for line in f:
        ss=line.strip().split('\t')
        if len(ss) == 3:
            answers.append(ss[2])

predicts=[]
with codecs.open(sys.argv[2],'r','utf-8') as f:
    for line in f:
        ss=line.strip()
        predicts.append(ss)

question_level=0
token_level=0
token_total=0

assert len(answers) == len(predicts)

for i in range(len(answers)):
    if answers[i] == predicts[i]:
        question_level+=1
    ans_tokens=answers[i].split()
    pred_tokens=predicts[i].split()

    min_l=min(len(ans_tokens),len(pred_tokens))

    for j in range(min_l):
        if ans_tokens[j] == pred_tokens[j]:
            token_level+=1

    token_total+=len(ans_tokens)

print("{}\t{}".format(question_level/len(answers)),token_level/token_total)



