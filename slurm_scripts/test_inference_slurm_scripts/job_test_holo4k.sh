#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="holo4k Inference"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate pytorch_env
python3 infer_test_set.py holo4k_dp holo4k/trained_model_m_hybrid_s_holo4k_ls_0.2_cw_0.8_1.2_1659108875.9171767/epoch_49
