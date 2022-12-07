#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu 
#SBATCH -t 4:00:00
#SBATCH --gpus=a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="coach420_mlig Inference"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

source ~/scratch/anaconda3/etc/profile.d/conda.sh
ml cuda/11.6.2
conda activate pytorch_env
sh reset_sasa.sh coach420_mlig
python3 infer_test_set.py coach420_mlig chen/trained_model_s_chen_m_gatv2_gl_12_kh_2_so_1669914883.3920903/cv_0/epoch_49 -m gatv2 -gl 12 -kh 2 -so
