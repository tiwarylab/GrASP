#!/bin/bash
#SBATCH -p RM-shared 
#SBATCH --ntasks-per-node=24
#SBATCH --time 12:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing SCPDB"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


ml anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
ml cuda/10.2
conda activate ob 
rm -r scPDB_data_dir/mol2/*
rm -r scPDB_data_dir/raw/*
rm -r scPDB_data_dir/processed/*
rm -r scPDB_data_dir/ready_to_parse_mol2/*
python3 parse_files.py train_p2rank
