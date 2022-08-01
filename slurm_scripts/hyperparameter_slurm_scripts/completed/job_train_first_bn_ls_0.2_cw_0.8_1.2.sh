#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="scPDB -m first_bn -ls 0.2 -cw 0.8 1.2"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate pytorch_env
python3 train.py -m first_bn -ls 0.2 -cw 0.8 1.2
