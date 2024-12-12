#!/bin/bash

source ../fast_prompt_alignment/util_scripts/init.sh ## ADD YOUR OWN PATH HERE

# Function to run the process
run() {
    local DATASET=$1
    local SUBDIR=$2
    local TEXT_MODEL=$3
    local OPTIM_MODE=$4
    local START_IDX=$5
    local END_IDX=$6
    local NUM_ITER=$7
    
    # Create Subdirectory
    OUTPUT_DATA_PATH=$(create_subdirectory $OUTPUT_BASE_DIR $SUBDIR)

    accelerate launch --main_process_port 11125 -m $INFERENCE_MODULE \
        --optimization_mode $OPTIM_MODE \
        --text_model_name $TEXT_MODEL \
        --dataset_name $DATASET \
        --eval_data_path $EVAL_DATA_PATH \
        --output_data_path $OUTPUT_DATA_PATH \
        --starting_point $START_IDX \
        --ending_point $END_IDX \
        --num_iter $NUM_ITER \
        --gpu_runs 0 \
        --openai_api_region us
}

# Function to run the process
score() {
    local DATASET=$1
    local SUBDIR=$2
    local TEXT_MODEL=$3
    local OPTIM_MODE=$4
    local START_IDX=$5
    local END_IDX=$6
    local SCORING_COLUMN=$7
    
    # Create Subdirectory
    OUTPUT_DATA_PATH=$(create_subdirectory $OUTPUT_BASE_DIR $SUBDIR)

    accelerate launch --main_process_port 11125 -m $INFERENCE_MODULE \
        --optimization_mode $OPTIM_MODE \
        --text_model_name $TEXT_MODEL \
        --dataset_name $DATASET \
        --eval_data_path $EVAL_DATA_PATH \
        --output_data_path $OUTPUT_DATA_PATH \
        --starting_point $START_IDX \
        --ending_point $END_IDX \
        --num_iter 1 \
        --gpu_runs 0 \
        --scoring_column $SCORING_COLUMN \
        --openai_api_region us
}

optimize_test_datasets() {
    local SUBDIR=$1
    local TEXT_MODEL=$2

    # Loop through data files
    for DATASET in "${datasets[@]}"; do
        # Execute the command
        run $DATASET $SUBDIR $TEXT_MODEL iterative 0 -1 1
    done
}