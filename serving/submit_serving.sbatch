#!/usr/bin/env bash
#SBATCH --job-name=qwen-serving
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=p01

# Submit from the workspace root after sourcing the serving environment.
set -euo pipefail

: "${SLURM_JOB_ID:?Use sbatch, not bash}"

cd -- "${SLURM_SUBMIT_DIR:?Submit from the workspace root}"

# Must match the GPU count requested above.
export TENSOR_PARALLEL_SIZE=1

echo "Serving node: $(hostname)"

exec srun --nodes=1 --ntasks=1 --gpu-bind=none --kill-on-bad-exit=1 \
    bash scripts/serving/run_serving.sh