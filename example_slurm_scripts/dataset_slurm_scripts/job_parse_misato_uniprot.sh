#!/bin/bash
#SBATCH -p RM-shared 
#SBATCH --ntasks-per-node=1
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing uniprot splits"


ml anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate clustering 
python3 uniprot_dfs.py -s misato
python3 uniprot_splits.py
