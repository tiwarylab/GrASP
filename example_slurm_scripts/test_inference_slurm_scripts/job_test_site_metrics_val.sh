#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=15
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection val"


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate clustering
python3 site_metrics.py val cv/trained_model_ag_multi_f_1_1688426705.5809176/cv_1/epoch_49
