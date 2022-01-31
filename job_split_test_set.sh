#!/bin/bash

## --- Uncomment for CPU ---
#SBATCH -p shared 
#SBATCH --ntasks-per-node=1
#SBATCH --time 2:00:00
#SBATCH --nodes=1
#SBATCH --job-name="move"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL


ml anaconda
ml cuda/10.2
conda activate ~/pytorch_env
python3 split_test_set.py 


