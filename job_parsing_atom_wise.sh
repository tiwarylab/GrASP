#!/bin/bash
#SBATCH -p shared 
#SBATCH --ntasks-per-node=24
#SBATCH --time 48:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing scPDB"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL


ml anaconda
ml cuda/10.2
conda activate ~/pytorch_env
python3 parsing_atoms_w_atom_feats_parallel_new_feats.py


