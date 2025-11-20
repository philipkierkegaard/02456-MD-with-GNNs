#!/bin/bash
#BSUB -q gpuv100
#BSUB -J painn_md17
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
##BSUB -u s234873@dtu.dk
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N 

module purge

module load cuda/12.1

ENV=/work3/s234873/02456-MD-with-GNNs/painn_env
source $ENV/bin/activate

export PATH=$ENV/bin:$PATH
export PYTHONPATH=$ENV/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"

# ---------------------------------------------
# 5) Run training
# ---------------------------------------------
python src/train.py