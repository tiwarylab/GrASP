#!/bin/bash
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=32
#SBATCH -t 16:00:00
#SBATCH --job-name="p2rank holo4k_chains"

ls benchmark_data_dir/holo4k_chains/split_pdb > p2rank_ds/holo4k_chains.ds
sed -i 's/^/..\/benchmark_data_dir\/holo4k_chains\/split_pdb\//' p2rank_ds/holo4k_chains.ds
../p2rank/distro/prank eval-predict p2rank_ds/holo4k_chains.ds -o test_metrics/holo4k_chains/p2rank -visualizations 0

ml anaconda3
conda activate clustering
python3 p2rank_site_metrics.py holo4k_chains
