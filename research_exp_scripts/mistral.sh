#!/bin/bash

# Imports
source ../fast_prompt_alignment/util_scripts/train.sh

# Model name
MODEL_SHORT="mistral"
MODEL_LONG="mistralai/Mistral-Nemo-Instruct-2407"

train $MODEL_SHORT $MODEL_LONG "hf" "original_descriptions_42k" "coco_captions" $OUTPUT_BASE_DIR