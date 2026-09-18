#!/usr/bin/env bash
# Copy to .env.sh, edit these literal values, then manually source it.
# Every source overwrites previous values. Paths below are specific to TP1.

# Input / Python environment
export PYTHON_BIN="/mnt/home/pohsuan_huang-sili-a1dca6/cl/scripts/training/envs/tp1/.venv/bin/python"
export MODEL="Qwen/Qwen3.5-9B"
export DATASET="/mnt/home/pohsuan_huang-sili-a1dca6/cl/dataset/mixed/siliconmind-retention-v1"

# Use a new output directory for each experiment.
export OUTPUT_DIR="/mnt/home/pohsuan_huang-sili-a1dca6/cl/outputs/experiment-001"

# W&B: none or wandb. Never store API keys here.
export REPORT_TO="none"
export WANDB_ENTITY="s96006730-siliconmind"
export WANDB_PROJECT="sm-dp-training"
export WANDB_NAME="experiment-001"

# TP1 toolchain patch: Triton's CUDA helper failed with inherited NVHPC nvc.
export CC="/cm/local/apps/gcc/14.2.0/bin/gcc"
export CXX="/cm/local/apps/gcc/14.2.0/bin/g++"
