#!/bin/bash

source ../fast_prompt_alignment/util_scripts/run.sh ## ADD YOUR OWN PATH HERE

# Loop through data files
for DATASET in "${datasets[@]}"; do
    # Execute the command
    run $DATASET scoring chatgpt scoring 0 -1 1
done
