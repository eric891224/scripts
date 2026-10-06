#!/usr/bin/env bash
# Copy to .env.sh. Edit the values, then source it manually.

# Serving executable
export VLLM_BIN="/mnt/home/pohsuan_huang-sili-a1dca6/cl/scripts/serving/envs/tp1/.venv/bin/vllm"

# Replace this example with the actual trained model directory.
export MODEL="/mnt/home/pohsuan_huang-sili-a1dca6/cl/outputs/experiment-001/final"

# Model ID exposed through the API
export SERVED_MODEL_NAME="rtl"

# Must match pf --local-port.
export PORT="8000"

# TP1 toolchain patch: Triton's CUDA helper failed with inherited NVHPC nvc.
export CC="/cm/local/apps/gcc/14.2.0/bin/gcc"
export CXX="/cm/local/apps/gcc/14.2.0/bin/g++"