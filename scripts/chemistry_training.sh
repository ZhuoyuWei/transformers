#!/bin/sh

python ../examples/run_chemistry_parser.py --data_dir ../../../data --output_dir ../../../output_test_7_bert_decoding --output_block_size=128 \
--do_train=True --per_gpu_train_batch_size=8 --do_evaluate=True --trained_checkpoints ../../../output_test_7_bert_decoding --decoding_type=pnt --num_train_epochs=$1 \
--encoder_lr=1e-3 --decoder_lr=1e-3 --encoder_warmup=0 --decoder_warmup=0


