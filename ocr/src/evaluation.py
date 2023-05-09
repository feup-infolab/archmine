import sys
import os
import cv2
import pytesseract
import pandas as pd
from functions import load_files_from_folder, load_images_from_folder, clean_text, match_dict
from strsimpy.weighted_levenshtein import WeightedLevenshtein
from sklearn.feature_extraction.text import CountVectorizer
from functions import substitution_cost, insertion_cost, deletion_cost
from algorithms import *

custom_config = r'-l por -c classify_enable_learning=0 classify_enable_adaptive_matcher=0'

def load_ground_truth_info(algorithm_name, images_path, gt_path):
    """Returns sample images and ground truth files with filenames"""
    print("Loading images and files...")
    sys.stdout.flush()
    if algorithm_name[0] == "baseline":
        images, filenames = load_images_from_folder(images_path, cv2.IMREAD_COLOR)
    else:
        images, filenames = load_images_from_folder(images_path, cv2.IMREAD_GRAYSCALE)
    gt_files, gt_filenames = load_files_from_folder(gt_path)
    print("Finished loading.\n")
    sys.stdout.flush()
    return images, filenames, gt_files, gt_filenames

def setup_algorithms(parameters_path, document_type, algorithms):
    """Returns algorithm's parameters' values"""
    algorithm_objects = []
    for method in algorithms:
        algorithm = Helper(method)
        if method != "baseline":
            df = pd.read_csv(parameters_path+method+".csv")
            row = df.loc[df['classification'] == document_type]
            list_parameters = row[algorithm.obj.get_parameters()].values[0]
            algorithm.obj.set_parameters(list_parameters)
        algorithm_objects.append(algorithm)
    return algorithm_objects

def save_measures_to_file(gt_files, gt_filenames, filenames, images, algorithm_objects, algorithm_name, folder):
    """Calculate and save results of character accuracy and F1-score in .csv files"""
    if os.path.isfile(dst_path+"/"+folder+"_char_accuracy.csv"):
        df_char_accuracy=pd.read_csv(dst_path+"/"+folder+"_char_accuracy.csv", index_col="file")
        df_f1_score=pd.read_csv(dst_path+"/"+folder+"_f1_score.csv", index_col="file")
    elif os.path.isfile(dst_path+"/char_accuracy.csv"):
        df_char_accuracy=pd.read_csv(dst_path+"/char_accuracy.csv", index_col="file")
        df_f1_score=pd.read_csv(dst_path+"/f1_score.csv", index_col="file")
    else:
        df_char_accuracy=pd.DataFrame(columns=[algorithm_name])
        df_f1_score=pd.DataFrame(columns=[algorithm_name])
        df_char_accuracy.index.name = "file"
        df_f1_score.index.name = "file"
    for index, gt_text in enumerate(gt_files):
        img_index = filenames.index(gt_filenames[index])
        img = images[img_index]
        for algorithm in algorithm_objects:
            if algorithm.obj.get_name() != "baseline":
                img = algorithm.obj.get_cv_method(img)
        print("Starting OCR task for digital representation "+gt_filenames[index]+"...")
        sys.stdout.flush()
        gt_text, text = apply_ocr(img, gt_text)
        print("Calculating measures...")
        sys.stdout.flush()
        character_accuracy, recall, precision = calculate_measures(gt_text, text)
        if precision == 0 or recall == 0:
            f1_score = 0
        else:
            f1_score = round(2*precision*recall/(precision+recall), 2)
        df_char_accuracy.loc[os.path.splitext(gt_filenames[index])[0], algorithm_name] = character_accuracy
        df_f1_score.loc[os.path.splitext(gt_filenames[index])[0], algorithm_name] = f1_score
    print("\nCharacter Accuracy results:\n")
    print(df_char_accuracy)
    sys.stdout.flush()
    print("\nF1-Score results:\n")
    print(df_f1_score)
    sys.stdout.flush()
    if folder != "global" and folder != "default":
        df_char_accuracy.to_csv(dst_path+"/"+folder+"_char_accuracy.csv")
        df_f1_score.to_csv(dst_path+"/"+folder+"_f1_score.csv")
    else:
        df_char_accuracy.to_csv(dst_path+"/char_accuracy.csv")
        df_f1_score.to_csv(dst_path+"/f1_score.csv")
    print("\nSaved results to .csv files.\n")
    sys.stdout.flush()

def apply_ocr(img, gt_text):
    """Returns normalized ground truth and textual OCR result"""
    text = pytesseract.image_to_string(img, config=custom_config)
    gt_text = ''.join(gt_text)
    gt_text = clean_text(gt_text)
    text = ''.join(text)
    text = clean_text(text)
    print("Finished OCR task.")
    sys.stdout.flush()
    return gt_text, text

def calculate_measures(gt_text, text):
    """Returns character accuracy, recall and precision measures"""
    if not text or len(text) <= 1:
        recall=0
        precision=0
    else:
        gt_vectorizer = CountVectorizer()
        gt_matrix = gt_vectorizer.fit_transform([gt_text])
        gt_matrix = gt_matrix.toarray()[0]
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None)
        matrix = vectorizer.fit_transform([text])
        matrix = matrix.toarray()[0]
        if not vectorizer.vocabulary_:
            recall=0
            precision=0
        else:
            count = match_dict(gt_vectorizer.vocabulary_, gt_matrix, vectorizer.vocabulary_, matrix)
            recall, precision = count/len(gt_matrix)*100, count/len(matrix)*100
    weighted_levenshtein = WeightedLevenshtein(
        substitution_cost_fn=substitution_cost,
        insertion_cost_fn=insertion_cost,
        deletion_cost_fn=deletion_cost)
    edit_distance = weighted_levenshtein.distance(text, gt_text)
    cer = min(len(gt_text), edit_distance)/len(gt_text)
    character_accuracy = round((1-cer)*100,2)
    print("Finished calculating measures.\n")
    sys.stdout.flush()
    return character_accuracy, recall, precision

"""Evaluate optimization of image processing algorithms in OCR using NSGA-II"""

try:
    gt_path = sys.argv[1]
    images_path=sys.argv[2]
    parameters_path=sys.argv[3]
    dst_path=sys.argv[4]
    document_type = sys.argv[5]
    if ',' in sys.argv[6]:
        algorithms = list(sys.argv[6][1:-1].split(","))
        algorithm_name = "+".join(algorithms)
    else:
        algorithm_name = sys.argv[6]
        algorithms = [algorithm_name]
except IndexError:
    print("Invalid command line arguments: 6 arguments are expected. Usage: python evaluation.py <ground truth path> <sample image path> <path to parameter values given by NSGA-II> <dst path for results of character accuracy and F1-score> <type of parameterization (default, global or a typology)> <preprocessing method>")
    sys.exit()

if not os.path.exists(dst_path):
    os.makedirs(dst_path)

images, filenames, gt_files, gt_filenames = load_ground_truth_info(algorithm_name, images_path, gt_path)

algorithm_objects = setup_algorithms(parameters_path, document_type, algorithms)

save_measures_to_file(gt_files, gt_filenames, filenames, images, algorithm_objects, algorithm_name, document_type)