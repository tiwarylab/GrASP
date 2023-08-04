#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="val Inference"

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/11.7.1
conda activate pytorch_env
#sh reset_sasa.sh cv
python3 infer_test_set.py cv/trained_model_ag_multi_f_1_1688426705.5809176/cv_1/epoch_49 -s val -ag multi -f 1
