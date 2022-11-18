#!/bin/bash
#set -x #print out all commands before executing
#set -e #abort bash script on error 
#SBATCH -e /home/mariacst/exoplanets/running/gNFW/Tcut650_NL/logs/error_BD1000_%A_%a.log
#SBATCH -o /home/mariacst/exoplanets/running/gNFW/Tcut650_NL/logs/output_BD1000_%A_%a.log
#SBATCH --array=1-100

IMG=/home/software/singularity/base-2022-05-20/

ex=T650_NL_gNFW_longerPriorG
N=1000
path=/home/mariacst/exoplanets/running/gNFW/Tcut650_NL/out/

mkdir ${path}/$SLURM_ARRAY_TASK_ID

for sigma in 0.1
do
    echo $sigma
    for rs in 5. 10.
    do
        echo $rs
        for gamma in 1.
        do
            echo $gamma
            singularity exec -B /home $IMG  python3 fitting_Ntimes.py $ex \
            $SLURM_ARRAY_TASK_ID $N $sigma $gamma $rs

            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma${sigma}_gamma${gamma}_rs${rs}*.iterinfo
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma${sigma}_gamma${gamma}_rs${rs}*v$SLURM_ARRAY_TASK_ID.txt
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma${sigma}_gamma${gamma}_rs${rs}*.points
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma${sigma}_gamma${gamma}_rs${rs}*.ptprob
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma${sigma}_gamma${gamma}_rs${rs}*ev.dat
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma${sigma}_gamma${gamma}_rs${rs}*stats.dat
            rm ${path}/$SLURM_ARRAY_TASK_ID/${ex}_N${N}_sigma${sigma}_gamma${gamma}_rs${rs}*resume.dat
        done
    done
done
