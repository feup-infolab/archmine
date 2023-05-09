#!/bin/bash

DST_PATH="../outputs/files/results/evaluation/default/"
ALGORITHMS=("baseline" "simple_threshold" "adaptive_threshold0" "adaptive_threshold1" "otsu_threshold" "triangle_threshold" "homogeneous_blur" "gaussian_blur" "median_blur" "bilateral_filtering" "erosion" "dilation" "opening" "closing" "morphological_gradient" "top_hat" "black_hat")
GROUND_TRUTH_TRANSCRIPTIONS="../data/evaluation_sample/transcriptions/"
GROUND_TRUTH_IMAGES="../data/evaluation_sample/images/"
PARAMETERS_PATH="../outputs/files/results/parameterization/default/"
SCRIPT="../src/evaluation.py"

for ALGORITHM in "${ALGORITHMS[@]}"
do
    echo "----------------------------------------------------------------------------------------------------"
    echo "Algorithm: $ALGORITHM"
    echo "----------------------------------------------------------------------------------------------------"
    python $SCRIPT $GROUND_TRUTH_TRANSCRIPTIONS $GROUND_TRUTH_IMAGES $PARAMETERS_PATH $DST_PATH "default" $ALGORITHM &&
    echo "----------------------------------------------------------------------------------------------------\n"
done