#!/bin/bash
#BSUB -q gpuv100
#BSUB -J painn_md17
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
##BSUB -u s234873@dtu.dk
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N 

# Load CUDA
module load cuda/12.1

# Path to environment
ENV=/work3/s234873/02456-MD-with-GNNs/painn_env

# Activate
source $ENV/bin/activate

# Run script using venv python
$ENV/bin/python src/train.py
