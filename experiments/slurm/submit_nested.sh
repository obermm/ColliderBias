#!/usr/bin/env bash
# One SLURM array task per line of experiments/nested_baselines/sweeps.txt.
#   sbatch --array=0-43 experiments/slurm/submit_nested.sh
#SBATCH --job-name=nested
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
python -u experiments/nested_baselines/run_nested.py --sweep "${SLURM_ARRAY_TASK_ID:-0}"
