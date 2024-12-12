#!/bin/bash

# Imports
source ../fast_prompt_alignment/util_scripts/run.sh ## ADD YOUR OWN PATH HERE

run "original_descriptions_42k" "train_data" "chatgpt" "iterative" 0 -1 10
