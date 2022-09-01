#!/bin/bash
#SBATCH -p RM-shared 
#SBATCH --ntasks-per-node=24
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --job-name="fetch uniprot"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


ml anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
ml cuda/10.2
echo "Fetching UniProts for $1"
python3 uniprot_dfs.py -s $1
