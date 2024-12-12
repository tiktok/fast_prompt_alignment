#!/bin/bash

# Imports
source ../fast_prompt_alignment/util_scripts/constants.sh ## ADD YOUR OWN PATH HERE
source ../fast_prompt_alignment/util_scripts/functions.sh ## ADD YOUR OWN PATH HERE
pip3 install diffusers -U
pip3 install openai
pip3 install torchmetrics -U
pip3 install transformers -U
pip3 install flash-attn -U
pip3 install timm
pip3 install av
pip3 install rotary_embedding_torch
pip3 install tiktoken
pip3 install transformers_stream_generator
pip3 install httpx==0.23.3
pip3 install numpy==1.26

export NCCL_SOCKET_IFNAME=eth0  # or the appropriate interface
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=360000  # Increase this value (default is 10 seconds)

cd ../fast_prompt_alignment/ ## ADD YOUR OWN PATH HERE

nohup python3 -m inference.utils &
nohup python3 -m inference.utils &
nohup python3 -m inference.utils &