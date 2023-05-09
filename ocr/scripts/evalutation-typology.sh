#!/bin/bash

DST_PATH="../outputs/files/results/evaluation/typology/"
ALGORITHMS=("baseline" "simple_threshold" "adaptive_threshold" "otsu_threshold" "triangle_threshold" "homogeneous_blur" "gaussian_blur" "median_blur" "bilateral_filtering" "erosion" "dilation" "opening" "closing" "morphological_gradient" "top_hat" "black_hat")
TYPOLOGIES=("structured_reports" "letters" "theatre_covers" "unstructured_reports" "process_covers")
GROUND_TRUTH_TRANSCRIPTIONS="../data/evaluation_sample/transcriptions/"
GROUND_TRUTH_IMAGES="../data/evaluation_sample/images/"
PARAMETERS_PATH="../outputs/files/results/parameterization/typology/"
SCRIPT="../src/evaluation.py"

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
        python $SCRIPT $GROUND_TRUTH_TRANSCRIPTIONS $GROUND_TRUTH_IMAGES $PARAMETERS_PATH $DST_PATH $TYPOLOGY $ALGORITHM &&
        echo "----------------------------------------------------------------------------------------------------\n"
    done
done