#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=15
#SBATCH -p RM
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection holo4k_mlig"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
conda activate clustering 
python3 site_metrics.py holo4k_mlig holo4k_mlig/trained_model_s_holo4k_mlig_m_gatv2_gl_12_kh_1_so_sp_5_3_1672394569.3117402/cv_0/epoch_49 -c louvain -d 5 -p .4 -a square
