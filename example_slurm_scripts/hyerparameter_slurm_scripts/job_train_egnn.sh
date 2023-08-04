#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="train_egnn"



module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/11.7.1
module load cudnn/8.0.4
conda activate pytorch_env
python3 train_egnn.py 


