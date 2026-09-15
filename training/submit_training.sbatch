#!/usr/bin/env bash
#SBATCH --job-name=qwen-sft-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Site-specific: pass --partition / --account / --qos / --constraint to sbatch.
# Select an H100 80GB partition/constraint; generic gpu:8 alone does NOT do so.
# Submit from workspace root, or export WORKSPACE_DIR as an absolute path.
# sbatch copies this file to a spool directory: do NOT derive paths from $0.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit with sbatch; do not execute this script on a login node." >&2
    exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" != 1 || "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "This launcher requires one node and one Slurm task (torchrun creates eight GPU workers)." >&2
    exit 2
fi

WORKSPACE_DIR="${WORKSPACE_DIR:-${SLURM_SUBMIT_DIR:?Submit from workspace root or set WORKSPACE_DIR}}"
cd -- "${WORKSPACE_DIR}"
WORKSPACE_DIR="$(pwd -P)"
if [[ ! -f "${WORKSPACE_DIR}/scripts/training/run_training.sh" ]]; then
    echo "WORKSPACE_DIR must contain scripts/training/run_training.sh" >&2
    exit 2
fi

# Load the cluster's CUDA/compiler modules or activate its environment HERE,
# if needed. No module names or container image are assumed by this template.
export PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_DIR}/sm-dp/.venv/bin/python}"
export NPROC_PER_NODE=8
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export DTYPE="${DTYPE:-bf16}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${WORKSPACE_DIR}/scripts/training/deepspeed_zero2.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/qwen-zero2-${SLURM_JOB_ID}}"
export WANDB_NAME="${WANDB_NAME:-qwen-zero2-${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Leave CUDA_VISIBLE_DEVICES under Slurm's control. Never overwrite it with 0..7.
# Read-only preflight; no model weights are loaded here.
"${PYTHON_BIN}" -c '
import torch
import deepspeed
assert torch.cuda.is_available(), "CUDA is unavailable in the selected Python environment"
assert torch.cuda.device_count() == 8, "Expected exactly eight visible GPUs"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "DeepSpeed:", deepspeed.__version__)
for index in range(8):
    device = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {device.name}, {device.total_memory / 2**30:.1f} GiB")
'

# Exactly ONE task sees the eight allocated GPUs; it starts ONE torchrun agent.
exec srun --nodes=1 --ntasks=1 --gpu-bind=none --kill-on-bad-exit=1 \
    bash "${WORKSPACE_DIR}/scripts/training/run_training.sh" "$@"
