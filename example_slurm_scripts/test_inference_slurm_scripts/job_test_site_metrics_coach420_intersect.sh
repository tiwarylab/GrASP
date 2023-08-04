#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=15
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection coach420_intersect"


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate clustering 
python3 site_metrics.py coach420_intersect coach420_mlig/trained_model_s_coach420_mlig_ag_multi_1672951806.257937/cv_0/epoch_49
