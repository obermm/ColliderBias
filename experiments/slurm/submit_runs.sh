#!/usr/bin/env bash
# One SLURM array task per line of `experiments/run.py --list` (cell, leg, optional --seed).
# RUN_GROUPS and LEGS narrow the list, as in run.py --groups / --legs.
#
#   sbatch --array=0-$(( $(python experiments/run.py --list --legs chain_ss,collider_ss,bpinn | wc -l) - 1 )) \
#          --export=ALL,LEGS=chain_ss,collider_ss,bpinn experiments/slurm/submit_runs.sh
#   # then, once the bpinn runs exist:
#   sbatch --array=0-3 --export=ALL,LEGS=chain_surrogate experiments/slurm/submit_runs.sh
#
# Wall times of the published runs (one GPU): PMMH 8 min (oscillator) to 9 h
# (particle); B-PINN 2-5 h (oscillator), 5-17 h (pendulum), 10-26 h (particle).
#SBATCH --job-name=colliderbias
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
LINE=$(python experiments/run.py --list ${RUN_GROUPS:+--groups "$RUN_GROUPS"} ${LEGS:+--legs "$LEGS"} \
       | sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p")
read -r CELL LEG EXTRA <<< "$LINE"
echo "task ${SLURM_ARRAY_TASK_ID:-0}: $CELL $LEG $EXTRA"
python -u experiments/run.py "$CELL" "$LEG" $EXTRA
