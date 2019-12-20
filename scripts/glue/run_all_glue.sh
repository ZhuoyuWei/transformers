#!/bin/sh

echo "RUN GLUE"

EXP_ID=$1
EXP_MODEL_NAME=$2 #['bert',...]
EXP_MODEL_PRETRAINED=$3 #['bert-large-uncased']
BATCH_SIZE=$4

#mkdir output dir
WDATA_DIR=/data/zhuoyu/glue/wdata
OUTPUT_DIR=$WDATA_DIR/$EXP_ID
mkdir $OUTPUT_DIR

#mkdir working dir
sudo apt-get install zip -y
sudo mkdir /zhuoyu_exp/
sudo chmod 777 /zhuoyu_exp
EXP_ROOT_DIR=/zhuoyu_exp/work
mkdir $EXP_ROOT_DIR
cd $EXP_ROOT_DIR

#copy data
DATA_DIR=$EXP_ROOT_DIR/data
mkdir $DATA_DIR
cp /data/zhuoyu/glue/data.zip $DATA_DIR/
cd $DATA_DIR
unzip data.zip
cd ..
GLUE_DATA_DIR=$DATA_DIR

#download code and install requirements
mkdir code
cd code
git clone https://github.com/ZhuoyuWei/transformers.git
cd transformers
sudo pip install -r requirements.txt
cd ../..


#run training/evaluation

cd code/transformers/
TASK_NAME=CoLA
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../


cd code/transformers/
TASK_NAME=SST-2
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../

cd code/transformers/
TASK_NAME=MRPC
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../


cd code/transformers/
TASK_NAME=STS-B
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../

cd code/transformers/
TASK_NAME=QQP
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../


cd code/transformers/
TASK_NAME=MNLI
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../

cd code/transformers/
TASK_NAME=QNLI
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../


cd code/transformers/
TASK_NAME=RTE
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../

cd code/transformers/
TASK_NAME=WNLI
python ./examples/run_glue.py \
    --model_type $EXP_MODEL_NAME \
    --model_name_or_path $EXP_MODEL_PRETRAINED \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DATA_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/
cd ../../

echo "Finish All"