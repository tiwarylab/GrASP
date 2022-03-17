#!/bin/bash
#SBATCH -p shared 
##SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing scPDB"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL


ml anaconda
ml cuda/10.2
conda activate ~/openbabel_env
python3 parse_benchmark_files.py train


