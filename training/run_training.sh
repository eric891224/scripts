#!/usr/bin/env bash
# Full fine-tuning (not LoRA). Run with bash; no cwd assumption.
# Example: MAX_STEPS=2 MAX_TRAIN_SAMPLES=32 bash run_training.sh
# Preview only: NPROC_PER_NODE=1 bash run_training.sh --dry-run --local-files-only
# CLI arguments at the end override environment/config settings.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Convenience default for direct invocation only; Slurm requires an explicit file.
export TRAINING_ENV_FILE="${TRAINING_ENV_FILE:-${SCRIPT_DIR}/envs/tp1/.env.sh}"
source "${SCRIPT_DIR}/load_env.sh"

args=(
    --model "${MODEL}"
    --dataset "${DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --max-steps "${MAX_STEPS}"
    --learning-rate "${LEARNING_RATE}"
    --batch-size "${BATCH_SIZE}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max-length "${MAX_LENGTH}"
    --dtype "${DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    --logging-steps "${LOGGING_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --seed "${SEED}"
    --dataset-num-proc "${DATASET_NUM_PROC}"
    --report-to "${REPORT_TO}"
)

if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
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
