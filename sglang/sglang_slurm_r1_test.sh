#!/bin/bash

# prevent OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deactivate
source /home/hungsh001/remijang/SGLANG/bin/activate
which python

set -x

ip=$(hostname --ip-address)

####################################################################################

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST="12.9"
# export RAY_DEDUP_LOGS=0
# export PYTHONUNBUFFERED=1
debug=0


cmd="python3 -m sglang.launch_server \
    --model-path /home/hungsh001/model_weights/DeepSeek-V3.1 \
    --tp 8 \
    --host 0.0.0.0 \
    --port 48763 \
    --served-model-name 'Pf. D' \
    --chunked-prefill-size 16384 \
    --max-running-requests 256 \
    --reasoning-parser deepseek-r1 \
    --grammar-backend "xgrammar"
    "

bash -c "$cmd" > log 2>&1
