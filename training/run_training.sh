#!/usr/bin/env bash

# Manually source your environment first. Operations: --dry-run, --smoke, --resume-from-checkpoint PATH.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${PYTHON_BIN:?Set PYTHON_BIN before running}"

for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        exec "$PYTHON_BIN" "$SCRIPT_DIR/training.py" --help
    fi
done

for setting in MODEL DATASET OUTPUT_DIR REPORT_TO; do
    if [[ -z "${!setting:-}" ]]; then
        echo "Missing ${setting}; manually source your environment first." >&2
        exit 2
    fi
done

if [[ "$REPORT_TO" == wandb ]]; then
    for setting in WANDB_ENTITY WANDB_PROJECT WANDB_NAME; do
        if [[ -z "${!setting:-}" ]]; then
            echo "${setting} is required when REPORT_TO=wandb" >&2
            exit 2
        fi
    done
elif [[ "$REPORT_TO" != none ]]; then
    echo "REPORT_TO must be none or wandb" >&2
    exit 2
fi

if [[ -n "${RANK:-}" || -n "${LOCAL_RANK:-}" ]]; then
    echo "Refusing nested torchrun; run this launcher once." >&2
    exit 2
fi

args=( "$SCRIPT_DIR/training.py" --model "$MODEL" --dataset "$DATASET"
       --output-dir "$OUTPUT_DIR" --report-to "$REPORT_TO" "$@" )

for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then
        exec "$PYTHON_BIN" "${args[@]}"
    fi
done

if [[ ! "${NPROC_PER_NODE:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set NPROC_PER_NODE to a positive integer matching the GPU allocation." >&2
    exit 2
fi

"$PYTHON_BIN" -c '
import os, sys, torch
workers, visible = int(sys.argv[1]), torch.cuda.device_count()
if visible < workers or (os.environ.get("SLURM_JOB_ID") and visible != workers):
    raise SystemExit(f"GPU/worker mismatch: visible={visible}, requested={workers}")
' "$NPROC_PER_NODE"

exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" --max-restarts=0 "${args[@]}"
