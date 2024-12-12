source ../fast_prompt_alignment/util_scripts/init.sh

DATASET="coco_captions"

python3 -m generation.generation \
    --model_name mistralai/Mistral-Nemo-Instruct-2407 \
    --model_path ${OUTPUT_BASE_DIR}finetune_mistral \
    --eval_data_path ../fast_prompt_alignment/data_prepped/${DATASET}_best_paraphrase.csv \
    --eval_data_column_name prompt \
    --output_data_path ${OUTPUT_BASE_DIR}${DATASET}_mistral_generated.csv