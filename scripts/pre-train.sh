#!/bin/bash

BATCH=20
today_date=$(TZ="America/Los_Angeles" date +"%m-%d-%y-%H%M") #

MODEL_NAME="byt5-small"
NAME="pretrain-OM-$today_date-$MODEL_NAME"

echo "Running the pre-training script with run_name $NAME"
#stoping training at 55000 results in a good pre-trained model, 
torchrun --nproc_per_node="$1" train.py \
    --run_name $NAME \
    --output_dir ../models/$NAME\
    --model_name_or_path google/byt5-small \
    --train_file ../data/pretraining/pre_train_paperversion_ontologies.jsonl \
    --per_device_train_batch_size $BATCH \
    --per_device_eval_batch_size $BATCH \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --max_target_length 512 \
    --max_source_length 512 \
    --save_steps 5000 \
    --mlm True \
    --mlm_mask_probability 0.1 \
    --mlm_max_mask_probability 0.35 \
    --mlm_mask_increase_epoch_end 5 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    --save_total_limit 20 \
    --do_train True\
    --mlm_mask_token "\x00" \
    --report_to all \

