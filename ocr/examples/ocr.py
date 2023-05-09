import pytesseract
import pandas as pd
import os
import cv2
import numpy as np
from functions import write_to_text_file
import sys

try:
    data = sys.argv[1]
    folder = sys.argv[2]
    dst_path = sys.argv[3]
except IndexError:
    print("Invalid command line arguments: 3 arguments are expected. Usage: python ocr.py <csv file with classified dataset> <dataset images' folder> <processed dataset files dest folder path>")
    sys.exit()

df = pd.read_csv(data)

def apply_image_preprocessing(filename, type):
    """Apply best image processing algorithms suited for each digital representation typology"""
    img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        if type == "process_cover":
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1, borderType=cv2.BORDER_REPLICATE)
        elif type == "theatre_cover":
            return cv2.adaptiveThreshold(img, 152, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 18)
        elif type == "letter":
            return cv2.bilateralFilter(4, 4, 31)
        elif type == "structured_report":
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8), iterations=9, borderType=cv2.BORDER_ISOLATED)
        elif type == "unstructured_report":
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1, borderType=cv2.BORDER_REFLECT_101)
        else:
            return cv2.imread(os.path.join(folder, filename))
    else:
        return None

for index, row in df.iterrows():
    filename = row['Name']
    img = apply_image_preprocessing(filename, row['Type'])
    if img is not None:
        custom_config = r'-l por -c classify_enable_learning=0 classify_enable_adaptive_matcher=0'
        extractedInformation = pytesseract.image_to_string(img, config=custom_config)
        text_filename = os.path.join(dst_path, filename[:filename.index(".")] + '.txt')
        write_to_text_file(text_filename, extractedInformation)
        print('Created ' + text_filename)
    else:
        print('Error! Image '+filename+' does not exist.')