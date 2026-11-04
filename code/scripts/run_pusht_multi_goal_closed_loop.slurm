#!/usr/bin/env bash
#SBATCH --job-name=jepa-multigoal-cem
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%2
#SBATCH --output=slurm-pusht-multigoal-cem-%A_%a.out
#SBATCH --error=slurm-pusht-multigoal-cem-%A_%a.err
#SBATCH --mail-user=junru@umd.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
seeds=(20261108 20261109 20261110 20261111 20261112)
seed="${seeds[${SLURM_ARRAY_TASK_ID:?}]}"
output="results/pusht_multi_goal_closed_loop/seed_${seed}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec .venv/bin/python code/experiments/run_pusht_multi_goal_closed_loop.py \
  --device cuda --seed "$seed" --output "$output"
