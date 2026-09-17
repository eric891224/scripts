#!/usr/bin/env bash
# Full fine-tuning (not LoRA). Run with bash; no cwd assumption.
# Example: MAX_STEPS=2 MAX_TRAIN_SAMPLES=32 bash run_training.sh
# Preview only: bash run_training.sh --dry-run --local-files-only
# CLI arguments at the end override the environment defaults below.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/envs/tp1/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
DATASET="${DATASET:-${WORKSPACE_DIR}/dataset/mixed/siliconmind-retention-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/qwen-domain-retention}"
# Optional override; otherwise use the template shipped with MODEL's tokenizer.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi

# W&B settings (read directly by the Trainer/W&B integration).
# Enable with REPORT_TO=wandb; keep disabled by default to avoid uploading logs.
# Authenticate with wandb login on the training machine. Never put API keys here.
REPORT_TO="${REPORT_TO:-none}"
export WANDB_ENTITY="${WANDB_ENTITY:-s96006730-siliconmind}"
export WANDB_PROJECT="${WANDB_PROJECT:-sm-dp-training}"
export WANDB_NAME="${WANDB_NAME:-qwen35-domain80-retention20-v1}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"  # No automatic model uploads.

args=(
    --model "${MODEL}"
    --dataset "${DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --epochs "${EPOCHS:-1}"
    --max-steps "${MAX_STEPS:--1}"
    --learning-rate "${LEARNING_RATE:-2e-5}"
    --batch-size "${BATCH_SIZE:-1}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
    --max-length "${MAX_LENGTH:-4096}"
    --dtype "${DTYPE:-bf16}"
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
    --logging-steps "${LOGGING_STEPS:-10}"
    --save-steps "${SAVE_STEPS:-250}"
    --save-total-limit "${SAVE_TOTAL_LIMIT:-2}"
    --seed "${SEED:-42}"
    --dataset-num-proc "${DATASET_NUM_PROC:-1}"
    --report-to "${REPORT_TO}"
)

if [[ -n "${CHAT_TEMPLATE}" ]]; then
    args+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
    args+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

# CUDA_VISIBLE_DEVICES is inherited, e.g. CUDA_VISIBLE_DEVICES=0 bash ...
if (( NPROC_PER_NODE > 1 )); then
    if [[ -n "${RANK:-}" || -n "${LOCAL_RANK:-}" ]]; then
        echo "Refusing nested torchrun: run this launcher once, not once per GPU rank." >&2
        exit 2
    fi
    # A single node only. Standalone rendezvous selects a free local port.
    exec "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" --max-restarts=0 \
        "${SCRIPT_DIR}/training.py" "${args[@]}" "$@"
fi
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/training.py" "${args[@]}" "$@"
