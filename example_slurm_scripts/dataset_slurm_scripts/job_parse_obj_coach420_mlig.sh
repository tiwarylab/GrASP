#!/bin/bash
#SBATCH -p RM
#SBATCH --ntasks-per-node=128
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing coach420_mlig surfaces"


ml anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate ob
python3 process_obj.py coach420_mlig
