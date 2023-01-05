#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="val Inference"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/11.7.1
conda activate pytorch_env
sh reset_sasa.sh cv
python3 infer_test_set.py val cv/trained_model_nn_0.1_1672886039.4995553/cv_0/epoch_49 
