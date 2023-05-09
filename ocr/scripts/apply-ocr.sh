#!/bin/bash

DATA="../outputs/files/data/classified_dataset.csv"
FOLDER="../outputs/images/dataset"
DST_PATH="../outputs/files/dataset/"
SCRIPT="../examples/ocr.py"

python $SCRIPT $DATA $FOLDER $DST_PATH