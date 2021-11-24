#!/bin/bash
## --- Uncomment for GPU ---
##SBATCH -p gpuk80
##SBATCH --gres=gpu:2
##SBATCH --ntasks-per-node=4
##SBATCH --time 3:00:00
##SBATCH --nodes=1
##SBATCH --job-name="scPDB GPU"

## --- Uncomment for CPU ---
#SBATCH -p shared 
#SBATCH --ntasks-per-node=6
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --job-name="scPDB CPU"


ml anaconda
ml cuda/10.2
conda activate ~/pytorch_env
python3 infer_test_set.py 


