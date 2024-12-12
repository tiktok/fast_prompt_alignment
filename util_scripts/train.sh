#!/bin/bash

source ../fast_prompt_alignment/util_scripts/init.sh

train_and_prepare_data() {
    local MODEL=$1
    local ONLINEMODEL=$2
    local FORMAT=$3
    local DATASET_NAME=$4
    local OUTPUT_BASE_DIR=$5

    if [ "$FORMAT" == "qwen" ]; then
        EXTENSION=".json"
    else
        EXTENSION=".csv"
    fi

    create_directory $TRAIN_DATA_DIR

    DATASET_OUTPUT_PATH="${TRAIN_DATA_DIR}${DATASET_NAME}${EXTENSION}"

    create_subdirectory $OUTPUT_BASE_DIR "finetune_${MODEL}"

    LOGPATH="${OUTPUT_BASE_DIR}finetune_${MODEL}"

    python3 -m $COMBINE_MODULE \
        --eval_data_path "" \
        --output_data_path "${OUTPUT_BASE_DIR}train_data/" \
        --dataset_name $DATASET_NAME

    python3 -m $FORMAT_MODULE \
        --input_file_path "${OUTPUT_BASE_DIR}train_data/${DATASET_NAME}_combined.csv" \
        --original_prompt_column "original_prompt" \
        --rephrased_prompt_column "best_paraphrase" \
        --output_file_path $DATASET_OUTPUT_PATH \
        --format $FORMAT

    torchrun --nproc_per_node=8 --master_port=11125 -m $TRAIN_MODULE \
        --model_name_or_path $ONLINEMODEL \
        --new_model_path $LOGPATH \
        --output_dir $LOGPATH-results \
        --new_model_name $MODEL \
        --trust_remote_code \
        --data_path $DATASET_OUTPUT_PATH \
        --num_train_epochs 25 \
        --bf16 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --max_grad_norm 1.0 \
        --learning_rate 5e-6 \
        --weight_decay 0.01 \
        --optim "paged_adamw_32bit" \
        --lr_scheduler_type "cosine" \
        --max_steps -1 \
        --warmup_ratio 0.03 \
        --group_by_length \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --max_seq_length 8192 \
        --report_to "wandb" \
        --greater_is_better False \
        --use_lora_target_modules
}

train() {
    local MODEL=$1
    local ONLINEMODEL=$2
    local FORMAT=$3
    local DATASET_NAME=$4
    local EVAL_DATASET_NAME=$5
    local OUTPUT_BASE_DIR=$6

    if [ "$FORMAT" == "qwen" ]; then
        EXTENSION=".json"
    else
        EXTENSION=".csv"
    fi

    DATASET_OUTPUT_PATH="${EVAL_DATA_PATH}${DATASET_NAME}${EXTENSION}"
    DATASET_EVAL_PATH="${EVAL_DATA_PATH}${EVAL_DATASET_NAME}${EXTENSION}"

    create_subdirectory $OUTPUT_BASE_DIR "finetune_${MODEL}"

    LOGPATH="${OUTPUT_BASE_DIR}finetune_${MODEL}"

    torchrun --nproc_per_node=8 --master_port=11125 -m $TRAIN_MODULE \
        --model_name_or_path $ONLINEMODEL \
        --new_model_path $LOGPATH \
        --output_dir $LOGPATH-results \
        --new_model_name $MODEL \
        --trust_remote_code \
        --data_path $DATASET_OUTPUT_PATH \
        --eval_data_path $DATASET_EVAL_PATH \
        --num_train_epochs 25 \
        --bf16 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --max_grad_norm 1.0 \
        --learning_rate 5e-6 \
        --weight_decay 0.01 \
        --optim "paged_adamw_32bit" \
        --lr_scheduler_type "cosine" \
        --max_steps -1 \
        --warmup_ratio 0.03 \
        --group_by_length \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --max_seq_length 8192 \
        --report_to "wandb" \
        --greater_is_better False \
        --use_lora_target_modules
}
