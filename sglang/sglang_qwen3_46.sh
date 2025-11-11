#!/bin/bash

# prevent OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deactivate
source .venv/bin/activate
which python

set -x

ip=$(hostname --ip-address)

####################################################################################

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.9" # for Ada Lovelace (RTX 4090, 4080, etc.)
# export RAY_DEDUP_LOGS=0
# export PYTHONUNBUFFERED=1
debug=0


cmd="python3 -m sglang.launch_server \
    --model-path /mnt/disk2/llm_team/Qwen3-30B-A3B-Thinking-2507 \
    --tp 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --chunked-prefill-size 16384 \
    --max-running-requests 256 \
    --reasoning-parser qwen3 \
    --grammar-backend "xgrammar" \
    "

bash -c "$cmd" > log 2>&1