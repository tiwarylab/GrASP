#!/bin/bash
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=1
#SBATCH --time 1:00:00
#SBATCH --nodes=1
#SBATCH --job-name="calculating coach420_mlig surfaces"


ml anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate pymol
python3 connolly_surface.py coach420_mlig -q 1
sbatch bridges_slurm_scripts/dataset_slurm_scripts/job_parse_obj_coach420_mlig.sh
