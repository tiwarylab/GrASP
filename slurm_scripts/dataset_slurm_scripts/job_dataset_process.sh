#!/bin/bash
#SBATCH -p shared 
#SBATCH --ntasks-per-node=16
#SBATCH --time 6:0:0
#SBATCH --nodes=1
#SBATCH --job-name="GASP Process"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL


ml anaconda
ml cuda/10.2
conda activate ~/pytorch_env
python3 dataset_process.py


