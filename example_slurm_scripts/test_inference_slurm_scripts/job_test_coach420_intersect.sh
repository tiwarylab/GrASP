#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 4:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="coach420_intersect Inference"

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/11.7.1
conda activate pytorch_env
sh reset_sasa.sh coach420_intersect
python3 infer_test_set.py coach420_mlig/trained_model_s_coach420_mlig_ag_multi_1672951806.257937/cv_0/epoch_49 -s coach420_intersect -ag multi
