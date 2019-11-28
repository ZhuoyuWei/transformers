#!/bin/sh

python ../examples/run_seq2seq.py --data_dir ../../../datas/data_ori --output_dir ../../../output_test_7_bert_origin \
--do_train=True --per_gpu_train_batch_size=8 --do_evaluate=True --trained_checkpoints ../../../output_test_7_bert_origin --num_train_epochs=50
