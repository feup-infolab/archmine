#!/bin/bash

DST_PATH="../outputs/files/results/parametrization/global/"
ALGORITHMS=("simple_threshold" "adaptive_threshold" "otsu_threshold" "triangle_threshold" "homogeneous_blur" "gaussian_blur" "median_blur" "bilateral_filtering" "erosion" "dilation" "opening" "closing" "morphological_gradient" "top_hat" "black_hat")
GROUND_TRUTH_TRANSCRIPTIONS="../data/parametrization_sample/transcriptions/"
GROUND_TRUTH_IMAGES="../data/parametrization_sample/images/"
SCRIPT="../src/genetic_algorithm.py"

for ALGORITHM in "${ALGORITHMS[@]}"
do
    echo "----------------------------------------------------------------------------------------------------"
    echo "Algorithm: $ALGORITHM"
    echo "----------------------------------------------------------------------------------------------------"
    python $SCRIPT $GROUND_TRUTH_TRANSCRIPTIONS $GROUND_TRUTH_IMAGES $DST_PATH 100 5 $ALGORITHM &&
    echo "----------------------------------------------------------------------------------------------------\n"
done