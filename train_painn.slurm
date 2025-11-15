#!/bin/bash
#SBATCH --job-name=painn-md17              # Job name
#SBATCH --output=painn_%j.out             # STDOUT file
#SBATCH --error=painn_%j.err              # STDERR file
#SBATCH --ntasks=1                         # 1 main process
#SBATCH --cpus-per-task=4                  # number of CPU cores
#SBATCH --gres=gpu:1                       # 1 GPU
#SBATCH --mem=16G                          # memory
#SBATCH --time=24:00:00                    # time limit (24 hours)
#SBATCH --partition=gpu                    # request GPU partition

echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "SLURM job ID: $SLURM_JOB_ID"

# Load modules (if needed)
module load cuda/12.1

# Activate your conda/venv
source /work3/$USER/02456-MD-with-GNNs/painn_env/bin/activate

# For safety: go to project directory
cd /work3/$USER/02456-MD-with-GNNs

# Run training
python3 src/train.py

echo "Job finished on $(date)"
