#!/bin/bash
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=24
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing coach420_intersect"


ml anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate ob
sh reset_dataset.sh coach420_intersect
python3 parse_files.py coach420_intersect
