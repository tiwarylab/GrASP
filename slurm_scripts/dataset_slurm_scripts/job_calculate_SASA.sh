#!/bin/bash
#SBATCH -p lrgmem 
#SBATCH --ntasks-per-node=48
#SBATCH --time 5:00:00
#SBATCH --nodes=1
#SBATCH --job-name="scPDB SASA"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL


ml anaconda
ml cuda/10.2
conda activate ~/pytorch_env
python3 calculate_ligand_duplicates_and_sasa.py


