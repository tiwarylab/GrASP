#!/bin/bash
#SBATCH -p RM-shared 
#SBATCH --ntasks-per-node=24
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing coach420"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


ml anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
ml cuda/10.2
conda activate ob
sh reset_dataset.sh coach420
python3 parse_files.py coach420
