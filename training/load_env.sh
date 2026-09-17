#!/usr/bin/env bash
# Sourced by both launchers. Configuration values belong only in TRAINING_ENV_FILE.
TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "${TRAINING_DIR}/../.." && pwd)"
if [[ -z "${TRAINING_ENV_FILE:-}" ]]; then
    echo "Set TRAINING_ENV_FILE to the path of your training config." >&2
    exit 2
fi
export TRAINING_ENV_FILE
if [[ ! -f "${TRAINING_ENV_FILE}" || ! -r "${TRAINING_ENV_FILE}" ]]; then
    echo "Training config not found or unreadable: ${TRAINING_ENV_FILE}" >&2
    echo "Check TRAINING_ENV_FILE and make the config readable on the compute node." >&2
    exit 2
fi
# Resolve once so srun and the child launcher use the same file regardless of cwd.
TRAINING_ENV_FILE="$(cd -- "$(dirname -- "${TRAINING_ENV_FILE}")" && pwd)/$(basename -- "${TRAINING_ENV_FILE}")"
source "${TRAINING_ENV_FILE}"

for training_setting in PYTHON_BIN MODEL DATASET OUTPUT_DIR NPROC_PER_NODE \
    BATCH_SIZE GRADIENT_ACCUMULATION_STEPS EPOCHS MAX_STEPS LEARNING_RATE \
    MAX_LENGTH DTYPE ATTN_IMPLEMENTATION LOGGING_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT \
    SEED DATASET_NUM_PROC REPORT_TO WANDB_ENTITY WANDB_PROJECT WANDB_NAME \
    WANDB_LOG_MODEL OMP_NUM_THREADS TOKENIZERS_PARALLELISM PYTHONUNBUFFERED; do
    if [[ -z "${!training_setting:-}" ]]; then
        echo "Missing required training setting ${training_setting} in ${TRAINING_ENV_FILE}" >&2
        exit 2
    fi
    export "${training_setting}"
done
unset training_setting
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi
