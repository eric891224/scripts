#!/usr/bin/env bash

export PYTHON_BIN="$(dirname -- "${BASH_SOURCE[0]}")/.venv/bin/python"
export MODEL="Qwen/Qwen3.5-9B"
export NPROC_PER_NODE=1
export DTYPE=bf16

# Optional: enable W&B logging
# REPORT_TO=wandb