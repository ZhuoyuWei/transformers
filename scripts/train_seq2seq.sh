#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ../examples/run_seq2seq.py --data_dir ../../../data --output_dir ../../../output_test_3_bert \
--do_train=True --per_gpu_train_batch_size=8 --do_evaluate=True --trained_checkpoints ../../../output_test_3_bert