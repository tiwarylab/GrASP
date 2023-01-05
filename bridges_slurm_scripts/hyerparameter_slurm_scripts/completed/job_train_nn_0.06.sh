#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 8:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="scPDB -nn 0.06"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/11.7.1
conda activate pytorch_env
python3 train.py -nn 0.06


