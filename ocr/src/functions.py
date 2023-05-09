import os
import cv2
import string
import re

def clean_text(text):
    """Normalize text"""
    # Remove text in brackets
    text = re.sub("[\[].*?[\]]", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Substitute all newlines with whitespaces
    text = text.replace('\n', ' ')
    # Remove extra whitespaces
    text = re.sub("\s\s+", " ", text)
    return text

def load_images_from_folder(folder, flag):
    """Load images and return them with corresponding filenames from a folder with Imread flag"""
    images = []
    filenames = []
    for (dirpath, dirnames, files) in os.walk(folder):
        for filename in files:
            filenames.append(filename.partition(".")[0])
            img = cv2.imread(os.path.join(dirpath,filename), flag)
            if img is not None:
                images.append(img)
    return images, filenames

def load_files_from_folder(folder):
    """Load files and return them with corresponding filenames from a folder"""
    textfiles = []
    filenames = []
    for (dirpath, dirnames, files) in os.walk(folder):
        for filename in files:
            filenames.append(filename.partition(".")[0])
            file = open(os.path.join(dirpath,filename), encoding='iso-8859-1')
            lines = file.readlines() 
            textfiles.append(lines)
    return textfiles, filenames

def write_to_text_file(filename, text):
    """Write to text file"""
    file = open(filename, "w", encoding='cp1252')
    file.write(text)
    file.close()

def insertion_cost(char):
    """Returns insertion cost for Levenshtein distance algorithm"""
    return 1.0

def deletion_cost(char):
    """Returns deletion cost for Levenshtein distance algorithm"""
    return 1.0

def substitution_cost(char_a, char_b):
    """Returns substitution cost for Levenshtein distance algorithm"""
    return 1.0

def match_dict(dict1, matrix1, dict2, matrix2):
    """Returns the number of matched words with the same number of occurrences between two dictionaries"""
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    shared_keys = dict1_keys.intersection(dict2_keys)
    count = 0
    for i in shared_keys:
        if matrix2[dict2.get(i)] == matrix1[dict1.get(i)]:
            count += 1
    return count