#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 4:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="holo4k_mlig Inference"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate pytorch_env
sh reset_sasa.sh holo4k_mlig
python3 infer_test_set.py holo4k_mlig train_full/trained_model_m_transformer_gn_s_train_full_sp_4_5_1664459350.690863/cv_0/epoch_49 -sp 4 5
