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
module load cuda/11.7.1
conda activate pytorch_env
sh reset_sasa.sh coach420_mlig
python3 infer_test_set.py coach420_mlig coach420_mlig/trained_model_s_coach420_mlig_m_gatv2_gl_12_kh_1_so_sp_5_3_1672393047.9643655/cv_0/epoch_49 -m gatv2 -gl 12 -kh 1 -so -sp 5 3
