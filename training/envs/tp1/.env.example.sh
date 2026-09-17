#!/usr/bin/env bash
# Copy to .env.sh; launchers source it automatically. Keep only trusted assignments
# here, never API keys. TRAINING_DIR and WORKSPACE_DIR are supplied by the loader.
# Precedence: Python CLI > existing environment > defaults in this file.

export PYTHON_BIN="${PYTHON_BIN:-${TRAINING_DIR}/envs/tp1/.venv/bin/python}"
export MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
export DATASET="${DATASET:-${WORKSPACE_DIR}/dataset/mixed/siliconmind-retention-v1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/qwen-zero2-${SLURM_JOB_ID:-local}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export DTYPE="${DTYPE:-bf16}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export SEED="${SEED:-42}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"

# Optional values preserve explicit empty overrides (e.g. DEEPSPEED_CONFIG=).
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-${TRAINING_DIR}/deepspeed_zero2.json}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE-}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES-}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT-}"

# Authenticate with wandb login; logging/model uploads stay disabled by default.
export REPORT_TO="${REPORT_TO:-none}"
export WANDB_ENTITY="${WANDB_ENTITY:-s96006730-siliconmind}"
export WANDB_PROJECT="${WANDB_PROJECT:-sm-dp-training}"
export WANDB_NAME="${WANDB_NAME:-qwen-zero2-${SLURM_JOB_ID:-local}}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
