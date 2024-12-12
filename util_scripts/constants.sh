# Folder paths
export EVAL_DATA_PATH="../fast_prompt_alignment/data_prepped/" ## ADD YOUR OWN PATH HERE
export OUTPUT_BASE_DIR="" ## ADD YOUR OWN PATH HERE
export TRAIN_DATA_DIR="${OUTPUT_BASE_DIR}data_prepped/"

# Array of data files
declare -a datasets=("coco_captions" "partiprompt")

# Modules
export INFERENCE_MODULE="inference.inference"
export TRAIN_MODULE="training.finetune"
export FORMAT_MODULE="training.format_input_data"
export COMBINE_MODULE="training.compile_existing_data"

# Mode
export ITER_ANNOT="iterative_for_annotation"