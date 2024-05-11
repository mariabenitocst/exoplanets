#!/bin/bash

IMG=/home/software/singularity/pytorch.simg
ex=example
N=10
path=out/
rank=20

for sigma in 0.01
do
    echo $sigma
    for C in 6.
    do
        echo $C
        for alpha in 1.
        do
            echo $alpha
            singularity exec -B /home $IMG  python3 fitting.py  $ex \
            $rank $N $sigma  $alpha $C
            singularity exec -B /home $IMG python3 fitting_with_no_DM.py $ex \
            $rank $N $sigma $alpha $C
            singularity exec -B /home $IMG python3 detection.py $ex \
            $rank $N $sigma $alpha $C
        done
    done
done
