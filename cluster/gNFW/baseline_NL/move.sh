#!/bin/bash

for rank in {1..100..1}
do
    echo ${rank}
    #mkdir /local/mariacst/2022_exoplanets/results/gNFW/baseline_NL_longer/${rank}/
    mv out/${rank}/baseline_NL_gNFW_longerPriorG_N1000_*post_equal_weights.dat /local/mariacst/2022_exoplanets/results/gNFW/baseline_NL_longer/${rank}/
done
