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
python3 infer_test_set.py holo4k_mlig holo4k_mlig/trained_model_m_transformer_gn_s_holo4k_mlig_1661869163.1372495/epoch_49
