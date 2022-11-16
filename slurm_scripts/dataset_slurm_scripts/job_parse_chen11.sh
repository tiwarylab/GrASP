#!/bin/bash
#SBATCH -p standard 
#SBATCH --ntasks-per-node=24
#SBATCH --time 4:00:00
#SBATCH --nodes=1
#SBATCH --job-name="parsing chen11"
#SBATCH --mail-user=zsmith7@umd.edu
#SBATCH --mail-type=ALL


source ~/scratch/anaconda3/etc/profile.d/conda.sh 
ml cuda/10.2.89
conda activate ob
sh reset_dataset.sh chen11
python3 parse_files.py chen11
