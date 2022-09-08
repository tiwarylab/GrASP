#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 4:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="coach420_mlig Inference"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate pytorch_env
sh reset_sasa.sh coach420_mlig
python3 infer_test_set.py coach420_mlig coach420_mlig/trained_model_m_transformer_gn_s_coach420_mlig_1661814669.7841315/epoch_49
