#!/bin/bash

#Batch Job Paremeters
#SBATCH --partition=SINICA_NTU
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1             # one torchrun per node https://stackoverflow.com/a/65897194
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node=8

# net
# export UCX_NET_DEVICES=mlx5_0:1
# export UCX_IB_GPU_DIRECT_RDMA=1

# enable NCCL log
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=3600

# prevent OOM
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deactivate
source /home/hungsh001/remijang/VLLM/bin/activate
which python

set -x

#####################################################################################

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export RAY_DEDUP_LOGS=0
# export PYTHONUNBUFFERED=1
export VLLM_USE_DEEP_GEMM=1
debug=0

cmd="vllm serve \
    /home/hungsh001/model_weights/DeepSeek-R1-0528 \
    --tensor-parallel-size 8 \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port 48763 \
    "

# srun --export=ALL --overlap --nodes=1 --ntasks=1 -w "$node_1" \
bash -c "$cmd" > log2 2>&1
