#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection coach420_mlig"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate ob 
python3 multisite_metrics_quantile_test.py coach420_mlig coach420_mlig/trained_model_m_transformer_gn_s_coach420_mlig_1661814669.7841315/epoch_49 -c louvain -d 5 -p .4 -a square -n 0
