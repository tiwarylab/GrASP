#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH -p standard
#SBATCH -t 1:00:00
#SBATCH --job-name="scPDB Metrics Community Detection holo4k_mlig"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


source ~/scratch/anaconda3/etc/profile.d/conda.sh
ml cuda/11.6.2
conda activate ob 
python3 multisite_metrics_quantile_test.py holo4k_mlig chen/trained_model_s_chen_m_gatv2_gl_12_kh_2_so_1669914883.3920903/cv_0/epoch_49 -c linkage -d 4 -p .4 -a square -su
