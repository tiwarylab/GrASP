#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=15
#SBATCH -p RM-small
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection coach420"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate ob 
python3 multisite_metrics_quantile_test.py coach420 coach420/trained_model_m_transformer_gn_s_coach420_1661810854.416886/epoch_49 -c louvain -d 4 -p .4 -a square
