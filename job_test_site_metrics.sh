#!/bin/bash
#SBATCH -p shared 
#SBATCH --ntasks-per-node=6
#SBATCH --time 12:00:00
#SBATCH --nodes=1
#SBATCH --job-name="scPDB Metrics"
#SBATCH --mail-user=strobelm@umd.edu
#SBATCH --mail-type=ALL


ml anaconda
ml cuda/10.2
conda activate ~/pytorch_env
##python3 site_metrics_closest_n.py 
## python3 site_metrics_closest_5.py 
python3 site_metrics.py 



