#!/usr/bin/env bash
#SBATCH --job-name=jepa-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec .venv/bin/python code/experiments/run_baseline.py "$@"
