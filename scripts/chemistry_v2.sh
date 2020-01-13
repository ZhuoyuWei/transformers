#!/bin/sh

python ../examples/run_chemistry_parser_v2.py --data_dir ../../../data --output_dir ../../../output_bert_decoding --output_block_size=128 \
--do_train=True --per_gpu_train_batch_size=8 --do_evaluate=False --trained_checkpoints ../../../output_bert_decoding --decoding_type=pnt --num_train_epochs=$1 \
--decoder_version=v2 --encoder_model_name_or_path=../../../bertmodels/encoder \
--decoder_model_name_or_path=../../../bertmodels/decoder --encoder_lr=$2 --decoder_lr=$2 --decoding_type=decoding
