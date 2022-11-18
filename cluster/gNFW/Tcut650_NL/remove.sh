#!/bin/bash

for rank in {1..100..1}
do
    echo ${rank}
    #ls out/${rank}/T650_NL_gNFW_longerPriorG_N1000_sigma0.3_rs10.0*.txt

    #rm out/${rank}/*
    rm out/${rank}/T650_NL_gNFW_longerPriorG_N1000_*ev.dat
    rm out/${rank}/T650_NL_gNFW_longerPriorG_N1000_*points
    rm out/${rank}/T650_NL_gNFW_longerPriorG_N1000_*ptprob
    rm out/${rank}/T650_NL_gNFW_longerPriorG_N1000_*txt
    rm out/${rank}/T650_NL_gNFW_longerPriorG_N1000_*iterinfo
done
