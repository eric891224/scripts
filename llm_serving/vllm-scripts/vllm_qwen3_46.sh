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


cmd="vllm serve \
    /mnt/disk2/llm_team/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser deepseek_r1"
    
bash -c "$cmd" > log 2>&1