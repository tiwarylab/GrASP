#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --job-name="p2rank coach"
  
        ../p2rank/distro/prank eval-predict p2rank_ds/coach420.ds -o test_metrics/coach420/p2rank -visualizations 0
	../p2rank/distro/prank eval-predict p2rank_ds/coach420\(mlig\).ds -o test_metrics/coach420_mlig/p2rank -visualizations 0
