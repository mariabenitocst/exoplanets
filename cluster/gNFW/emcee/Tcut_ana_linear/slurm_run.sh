#!/bin/bash
#set -x #print out all commands before executing
#set -e #abort bash script on error
#SBATCH --job-name=BD_sig0.3
#SBATCH --time=2-00:00:00
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH -e /home/sven/repos/exoplanets/logs/slurm_run_sig0.3/error_BD100_%A_%a.log
#SBATCH -o /home/sven/repos/exoplanets/logs/slurm_run_sig0.3/output_BD100_%A_%a.log
#SBATCH --array=1-100

source /home/sven/exoplanetenv/bin/activate

env

ex=fixedT10v100
N=100000
sigma=0.3

python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.3 20.
echo "Finished script 1."

python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.4 20.
echo "Finished script 2."

python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.5 20.
echo "Finished script 3."






