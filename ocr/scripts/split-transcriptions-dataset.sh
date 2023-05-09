#!/bin/bash

SCRIPT="../src/split_transcriptions_dataset.py"
TYPOLOGIES=("structured_reports" "letters" "theatre_covers" "unstructured_reports" "process_covers")

for TYPOLOGY in "${TYPOLOGIES[@]}"
do
    echo "----------------------------------------------------------------------------------------------------"
    echo "Splitting sample for $TYPOLOGY..."
    echo "----------------------------------------------------------------------------------------------------"
    IMAGES_SAMPLE_PATH="../data/sample/images/$TYPOLOGY/"
    TRANSCRIPTIONS_SAMPLE_PATH="../data/sample/transcriptions/$TYPOLOGY/"
    IMAGES_PARAMETERIZATION_PATH="../data/parametrization_sample/images/$TYPOLOGY/"
    IMAGES_EVALUATION_PATH="../data/evaluation_sample/images/$TYPOLOGY/"
    TRANSCRIPTIONS_PARAMETERIZATION_PATH="../data/parametrization_sample/transcriptions/$TYPOLOGY/"
    TRANSCRIPTIONS_EVALUATION_PATH="../data/evaluation_sample/transcriptions/$TYPOLOGY/"
    python $SCRIPT $IMAGES_SAMPLE_PATH $TRANSCRIPTIONS_SAMPLE_PATH $IMAGES_PARAMETERIZATION_PATH $IMAGES_EVALUATION_PATH $TRANSCRIPTIONS_PARAMETERIZATION_PATH $TRANSCRIPTIONS_EVALUATION_PATH
    echo "----------------------------------------------------------------------------------------------------\n"
done