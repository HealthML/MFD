#!/bin/bash -e

#SBATCH --partition=gpupro,gpua100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128gb
#SBATCH --time=3-00:00:00
#SBATCH -o slurm_logs/%j_%x_%A_%a.log
#SBATCH --job-name=runsweep

echo "${SLURM_JOB_ID}"
echo "${SLURM_ARRAY_JOB_ID:-''}"
echo "${SLURM_ARRAY_TASK_ID:-''}"

# Initialize conda:
eval "$(conda shell.bash hook)"
conda activate nucleotran_cuda11_2

# alternatively, to create from config:
python src/train_wandb.py --sweep_id "${1}" --n_runs=1

