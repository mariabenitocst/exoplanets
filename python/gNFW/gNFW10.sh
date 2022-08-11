#!/bin/bash
#set -x #print out all commands before executing
#set -e #abort bash script on error 
#SBATCH -e /home/mariacst/exoplanets/exoplanets/python/gNFW/logs/error_gNFW_baseline_10_%A_%a.log
#SBATCH -o /home/mariacst/exoplanets/exoplanets/python/gNFW/logs/output_gNFW_baseline_10_%A_%a.log
#SBATCH --array=21-200

source /home/mariacst/exoplanets/.venv/bin/activate

python3 experimental_sensitivity_CL_gNFW_slurm.py 10. $SLURM_ARRAY_TASK_ID -0.5 0.3
