#!/bin/bash
#set -x #print out all commands before executing
#set -e #abort bash script on error 
#SBATCH -e /home/mariacst/exoplanets/exoplanets/python/gNFW/logs/error_gNFW_T650_5_%A_%a.log
#SBATCH -o /home/mariacst/exoplanets/exoplanets/python/gNFW/logs/output_gNFW_T650_5_%A_%a.log
#SBATCH --array=1-15

source /home/mariacst/exoplanets/.venv/bin/activate

python3 experimental_sensitivity_CL_gNFW_slurm.py 5. $SLURM_ARRAY_TASK_ID 0.5 1.3
