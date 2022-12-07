#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu 
#SBATCH -t 4:00:00
#SBATCH --gpus=a100:4
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="chen -m gatv2 -gl 12 -kh 1 -so"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


rm benchmark_data_dir/chen11/processed/*
rm benchmark_data_dir/joined/processed/*
source ~/scratch/anaconda3/etc/profile.d/conda.sh
ml cuda/11.6.2
conda activate pytorch_env
python3 train.py -s chen -m gatv2 -gl 12 -kh 1 -so


