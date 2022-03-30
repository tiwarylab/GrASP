#!/bin/bash
## --- Uncomment for GPU ---
#SBATCH -p gpuk80
#SBATCH --gres gpu:4
#SBATCH --ntasks-per-node=8
#SBATCH --time 5:00:00
#SBATCH --nodes=1
#SBATCH --job-name="scPDB GPU"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL

## --- Uncomment for CPU ---
##SBATCH -p shared 
##SBATCH --ntasks-per-node=6
##SBATCH --time 0:30:00
##SBATCH --nodes=1
##SBATCH --job-name="scPDB CPU GRU Debug"
##SBATCH --mail-user=strobelm@umd.edu
##SBATCH --mail-type=ALL


ml anaconda
ml cuda/10.2
conda activate ~/pytorch_env
python3 train.py 0.02


