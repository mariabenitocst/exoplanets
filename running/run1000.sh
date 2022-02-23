#!/bin/bash
#SBATCH --partition=short
#SBATCH --array=1-40
#SBATCH -e /home/mariacst/exoplanets/running/logs/error_fixedT10Tcut650_nocutTwn_nWalkers0_BD1000_%A_%a.log
#SBATCH -o /home/mariacst/exoplanets/running/logs/output_fixedT10Tcut650_nocutTwn_nWalkers0_BD1000_%A_%a.log
source /home/mariacst/exoplanets/running/.env/bin/activate

ex=fixedT10Tcut650_nocutTwn_nWalkers0
N=1000
sigma=0.2

python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.2 20.
