#!/bin/bash

DST_PATH="../outputs/files/results/parametrization/typology/"
ALGORITHMS=("simple_threshold" "adaptive_threshold" "otsu_threshold" "triangle_threshold" "homogeneous_blur" "gaussian_blur" "median_blur" "bilateral_filtering" "erosion" "dilation" "opening" "closing" "morphological_gradient" "top_hat" "black_hat")
TYPOLOGIES=("structured_reports" "letters" "theatre_covers" "unstructured_reports" "process_covers")
SCRIPT="../src/genetic_algorithm.py"

for ALGORITHM in "${ALGORITHMS[@]}"
do
    for TYPOLOGY in "${TYPOLOGIES[@]}"
    do
        echo "----------------------------------------------------------------------------------------------------"
        echo "Digital Representation Type: $TYPOLOGY"
        echo "----------------------------------------------------------------------------------------------------"
        echo "----------------------------------------------------------------------------------------------------"
        echo "Algorithm: $ALGORITHM"
        echo "----------------------------------------------------------------------------------------------------"
        python $SCRIPT "../data/parametrization_sample/transcriptions/$TYPOLOGY/" "../data/parametrization_sample/images/$TYPOLOGY/" $DST_PATH 100 5 $ALGORITHM &&
        echo "----------------------------------------------------------------------------------------------------\n"
    done
done