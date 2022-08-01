#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="coach420 Inference"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate pytorch_env
python3 infer_test_set.py coach420_dp coach420/trained_model_m_hybrid_s_coach420_ls_0.2_cw_0.8_1.2_1659108631.326054/epoch_49
