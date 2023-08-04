#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 4:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="holo4k_chains Inference"

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/11.7.1
conda activate pytorch_env
sh reset_sasa.sh holo4k_chains
python3 infer_test_set.py holo4k_mlig/trained_model_s_holo4k_mlig_ag_multi_1672951807.304454/cv_0/epoch_49 -s holo4k_chains -ag multi
