#!/bin/sh
#BSUB -q gpuv100
#BSUB -J painn_md17
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 10GB
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load cuda/12.1
source /work3/s234873/02456-MD-with-GNNs/painn_env/bin/activate

python3 src/train.py
