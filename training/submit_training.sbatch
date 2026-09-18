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
#SBATCH --partition=p01

# Submit from workspace root after manually sourcing your environment.
set -euo pipefail

: "${SLURM_JOB_ID:?Use sbatch, not bash}"

if [[ "${SLURM_JOB_NUM_NODES:-}" != 1 || "${SLURM_NTASKS:-}" != 1 ]]; then
    echo "This script requires one node and one Slurm task." >&2
    exit 2
fi

cd -- "${SLURM_SUBMIT_DIR:?Submit from workspace root}"

# Must match the GPU request above; never overwrite CUDA_VISIBLE_DEVICES.
export NPROC_PER_NODE=8

exec srun --nodes=1 --ntasks=1 --gpu-bind=none --kill-on-bad-exit=1 \
    bash scripts/training/run_training.sh "$@"
