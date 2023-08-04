#!/bin/bash
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=32
#SBATCH -t 16:00:00
#SBATCH --job-name="p2rank holo"
  
        ../p2rank/distro/prank eval-predict p2rank_ds/holo4k.ds -o test_metrics/holo4k/p2rank -visualizations 0
