#!/bin/bash

for rank in {1..100..1}
do
    echo ${rank}
    #rm out/${rank}/*
    rm out/${rank}/*points
    rm out/${rank}/*resume.dat
    rm out/${rank}/*stats.dat
    #rm out/${rank}/baseline_NL_gNFW_longerPriorG_N100_sigma0.3_gamma1.0_*points
    rm out/${rank}/*ev.dat
    rm out/${rank}/*ptprob
    rm out/${rank}/*txt
    rm out/${rank}/*iterinfo
done
