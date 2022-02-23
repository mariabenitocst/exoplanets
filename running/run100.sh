#!/bin/bash
#SBATCH --array=15
#SBATCH -e /home/mariacst/exoplanets/running/logs/error_fixedT10Tcut650_nocutTwn_nWalkers4_BD100_%A_%a.log
#SBATCH -o /home/mariacst/exoplanets/running/logs/output_fixedT10Tcut650_nocutTwn_nWalkers4_BD100_%A_%a.log
source /home/mariacst/exoplanets/running/.env/bin/activate

ex=fixedT10Tcut650_nocutTwn_nWalkers4
N=100
sigma=0.2

python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.2 20.
