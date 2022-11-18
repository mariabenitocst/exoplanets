#!/bin/bash

for rank in {1..100..1}
do
    echo ${rank}
    #mkdir /local/mariacst/2022_exoplanets/results/gNFW/Tcut650_NL_longer/${rank}/
    mv out/${rank}/T650_NL_gNFW_longerPriorG_N1000_*post_equal_weights.dat /local/mariacst/2022_exoplanets/results/gNFW/Tcut650_NL_longer/${rank}/

done
