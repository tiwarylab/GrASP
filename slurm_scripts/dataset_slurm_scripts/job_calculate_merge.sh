#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH -p shared
#SBATCH -t 8:00:00
#SBATCH --job-name="scPDB merge and parse"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL


module load anaconda
module load cuda/10.2
conda activate ~/openbabel_env
python3 merge.py


