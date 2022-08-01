#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=15
#SBATCH -p RM-small
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection holo4k"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate ob 
python3 multisite_metrics_quantile_test.py holo4k_dp holo4k/trained_model_m_hybrid_s_holo4k_ls_0.2_cw_0.8_1.2_1659108875.9171767/epoch_49
 

