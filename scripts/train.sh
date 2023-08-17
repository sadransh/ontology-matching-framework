#!/bin/bash

BATCH=20
today_date=$(TZ="America/Los_Angeles" date +"%m-%d-%y-%H%M") #

MODEL_NAME="checkpoint-60000"
NAME="fine-tune-OM-$today_date-form-$MODEL_NAME"

echo "Running the training script with run_name $NAME"

#around 20 epochs model reaches a SOTA state.

torchrun --nproc_per_node="$1" train.py \
    --run_name $NAME \
    --seed 43\
    --output_dir ../models/$NAME\
    --model_name_or_path ../models/pre-trained/checkpoint-55000 \
    --train_file ../data/finetuning/07-26-23_pharmbody_upsampled/all_finetuning_train_tree_pharmbodyup_fixed.jsonl \
    --validation_file ../data/finetuning/07-26-23_pharmbody_upsampled/all_finetuning_eval_tree_pharmbodyup_fixed.jsonl \
    --per_device_train_batch_size $BATCH \
    --per_device_eval_batch_size $BATCH \
    --num_train_epochs 50 \
    --logging_steps 50 \
    --max_target_length 512 \
    --max_source_length 512 \
    --save_steps 5000 \
    --mlm False \
    --mlm_mask_probability 0.1 \
    --mlm_max_mask_probability 0.35 \
    --mlm_mask_increase_epoch_end 5 \
    --learning_rate 0.001 \
    --warmup_ratio 0.03 \
    --save_total_limit 20 \
    --do_train True\
    --do_eval False\
    --report_to all \

# --eval_steps 5000 \
# --evaluation_strategy steps \
