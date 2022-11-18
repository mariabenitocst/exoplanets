#!/bin/bash
#set -x #print out all commands before executing
#set -e #abort bash script on error 
#SBATCH -e /home/mariacst/exoplanets/running/power_law/baseline_NL/logs/error_BD1000_%A_%a.log
#SBATCH -o /home/mariacst/exoplanets/running/power_law/baseline_NL/logs/output_BD1000_%A_%a.log
#SBATCH --array=1-200

IMG=/home/software/singularity/base-2022-05-20/

ex=baseline_NL
N=1000
path=/home/mariacst/exoplanets/running/power_law/baseline_NL/out/

mkdir ${path}/$SLURM_ARRAY_TASK_ID

for sigma in 0.3
do
    echo $sigma
    for rs in 5.
    do
        echo $rs
        for gamma in 0.5 0.6 0.7 0.8 0.9 1.
        do
            echo $gamma
            singularity exec -B /home $IMG  python3 fitting_Ntimes.py $ex \
            $SLURM_ARRAY_TASK_ID $N $sigma $gamma $rs

            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma0.3_gamma${gamma}_rs${rs}*.iterinfo
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma0.3_gamma${gamma}_rs${rs}*v$SLURM_ARRAY_TASK_ID.txt
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma0.3_gamma${gamma}_rs${rs}*.points
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma0.3_gamma${gamma}_rs${rs}*.ptprob
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma0.3_gamma${gamma}_rs${rs}*ev.dat
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma0.3_gamma${gamma}_rs${rs}*stats.dat
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma0.3_gamma${gamma}_rs${rs}*resume.dat
        done
    done
done
