#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=15
#SBATCH -p RM
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection val"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate clustering
python3 site_metrics.py val cv/trained_model_ag_multi_1672951703.4392676/cv_0/epoch_49
