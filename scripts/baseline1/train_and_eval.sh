#!/bin/sh

EXP_ID=$1
TRAIN_EPOCH=$2
LR=$3
WDATA_DIR=/data/zhuoyu/semantic_parsing/wdata
OUTPUT_DIR=$WDATA_DIR/$EXP_ID
mkdir $OUTPUT_DIR

sudo apt-get install zip -y

sudo mkdir /zhuoyu_exp/
sudo chmod 777 /zhuoyu_exp

EXP_ROOT_DIR=/zhuoyu_exp/work
mkdir $EXP_ROOT_DIR
cd $EXP_ROOT_DIR

#CODE
CODE_DIR=$EXP_ROOT_DIR/code
mkdir $CODE_DIR
cd $CODE_DIR
git clone https://github.com/ZhuoyuWei/transformers.git
cd transformers
git checkout chemistry_v4
sudo pip install -r requirements.txt
cd $EXP_ROOT_DIR
SCRIPT_DIR=$CODE_DIR/transformers/scripts/chemistry_parser_eval
echo $SCRIPT_DIR

#DATA
DATA_DIR=$EXP_ROOT_DIR/data
cp -r /data/zhuoyu/semantic_parsing/data/ $EXP_ROOT_DIR/


#MODEL
cp -r /data/zhuoyu/semantic_parsing/models $EXP_ROOT_DIR/
MODEL_DIR=$EXP_ROOT_DIR/models


#RUNNING
cd $SCRIPT_DIR
python ../../examples/run_chemistry_parser_v2.py --data_dir $DATA_DIR/ --output_dir $OUTPUT_DIR/bert_output --output_block_size=128 \
    --do_train=True --per_gpu_train_batch_size=8 --do_evaluate=True  --num_train_epochs=$TRAIN_EPOCH \
    --decoder_version=v2 --encoder_model_name_or_path=$MODEL_DIR/encoder \
    --decoder_model_name_or_path=$MODEL_DIR/decoder --encoder_lr=$LR --decoder_lr=$LR \
    --decoding_type=decoding --trained_checkpoints=$OUTPUT_DIR/bert_output

python ../../examples/run_chemistry_parser.py --data_dir $DATA_DIR/ --output_dir $OUTPUT_DIR/bert_output --output_block_size=128 \
--do_train=True --per_gpu_train_batch_size=8 --do_evaluate=True --trained_checkpoints $OUTPUT_DIR/bert_output --decoding_type=decoding --num_train_epochs=$TRAIN_EPOCH
--encoder_lr=$LR  --decoder_lr=$LR
python ../chemistry_parser_eval/scorer.py $DATA_DIR/train.tsv $OUTPUT_DIR/bert_output/dev.res >> $OUTPUT_DIR/log


