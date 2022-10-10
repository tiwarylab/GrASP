#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 16:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="scPDB -m transformer_gn -sp 4 5 -cw 10 percent"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL

## train.py args node_noise_variance, ['cv', 'train_full', 'chen', 'coach420', 'holo4k', 'sc6k']

module load anaconda3
conda activate # source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load cuda/10.2
conda activate pytorch_env
python3 train.py -m transformer_gn -sp 4 5 -cw 0.92 5


