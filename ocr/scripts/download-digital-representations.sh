#!/bin/bash

SRC_FILES="../data/*.csv"
DST_FOLDER="../outputs/images/dataset"
SCRIPT="../src/download_digital_representations.py"

for file in $SRC_FILES
do
    python $SCRIPT $file $DST_FOLDER
done