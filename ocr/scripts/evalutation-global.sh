#!/bin/bash

DST_PATH="../outputs/files/results/evaluation/global/"
ALGORITHMS=("baseline" "simple_threshold" "adaptive_threshold" "otsu_threshold" "triangle_threshold" "homogeneous_blur" "gaussian_blur" "median_blur" "bilateral_filtering" "erosion" "dilation" "opening" "closing" "morphological_gradient" "top_hat" "black_hat")
GROUND_TRUTH_TRANSCRIPTIONS="../data/evaluation_sample/transcriptions/"
GROUND_TRUTH_IMAGES="../data/evaluation_sample/images/"
PARAMETERS_PATH="../outputs/files/results/parameterization/global/"
SCRIPT="../src/evaluation.py"

for ALGORITHM in "${ALGORITHMS[@]}"
do
    echo "----------------------------------------------------------------------------------------------------"
    echo "Algorithm: $ALGORITHM"
    echo "----------------------------------------------------------------------------------------------------"
    python $SCRIPT $GROUND_TRUTH_TRANSCRIPTIONS $GROUND_TRUTH_IMAGES $PARAMETERS_PATH $DST_PATH "global" $ALGORITHM &&
    echo "----------------------------------------------------------------------------------------------------\n"
done