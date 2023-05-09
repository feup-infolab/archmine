import numpy as np
import cv2
import pytesseract
import sys
import os
import pandas as pd
from pymoo.optimize import minimize
from algorithms import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from strsimpy.weighted_levenshtein import WeightedLevenshtein
from sklearn.feature_extraction.text import CountVectorizer
from functions import substitution_cost, insertion_cost, deletion_cost, load_images_from_folder, load_files_from_folder, clean_text, match_dict
from multiprocessing.pool import ThreadPool

def call_method(method):
    """Calls functions to return information about the image processing methods' variables and constraints"""
    switcher = {
        "simple_threshold": simple_threshold,
        "adaptive_threshold": adaptive_threshold,
        "otsu_threshold": otsu_threshold,
        "triangle_threshold": triangle_threshold,
        "bilateral_filtering": bilateral_filtering,
        "homogeneous_blur": homogeneous_blur,
        "median_blur": median_blur,
        "gaussian_blur": gaussian_blur,
        "erosion": erosion,
        "dilation": dilation,
        "opening": opening,
        "closing": closing,
        "morphological_gradient": morphological_gradient,
        "top_hat": top_hat,
        "black_hat": black_hat
    }
    func = switcher.get(method, "")
    return func()

def apply_ocr(X):
    """Returns textual OCR results"""
    results=[]
    selected_algorithm.set_parameters(X)
    text_files=[]
    for img in images:
        text = pytesseract.image_to_string(selected_algorithm.get_cv_method(img), config=custom_config)
        text_files.append(text)
    results.append(text_files)
    return results

def simple_threshold():
    """Returns the number of variables, constraints, lower, upper and variable names for Simple Thresholding"""
    return SimpleThresholdAlgorithm()

def adaptive_threshold():
    """Returns the number of variables, limits, constraints, and variable names for Adaptive Thresholding"""
    return AdaptiveThresholdAlgorithm()

def otsu_threshold():
    """Returns the number of variables, limits, constraints, and variable names for Otsu Thresholding"""
    return OtsuThresholdAlgorithm()

def triangle_threshold():
    """Returns the number of variables, limits, constraints, and variable names for Triangle Thresholding"""
    return TriangleThresholdAlgorithm()

def bilateral_filtering():
    """Returns the number of variables, limits, constraints, and variable names for Bilateral Filter"""
    return BilateralFilteringAlgorithm()

def homogeneous_blur():
    """Returns the number of variables, limits, constraints, and variable names for Homogeneous Blur"""
    return HomogeneousBlurAlgorithm()

def median_blur():
    """Returns the number of variables, limits, constraints, and variable names for Median Blur"""
    return MedianBlurAlgorithm()

def gaussian_blur():
    """Returns the number of variables, limits, constraints, and variable names for Gaussian Blur"""
    return GaussianBlurAlgorithm()

def erosion():
    """Returns the number of variables, limits, constraints, and variable names for Erosion"""
    return ErosionAlgorithm()

def dilation():
    """Returns the number of variables, limits, constraints, and variable names for Dilation"""
    return DilationAlgorithm()

def opening():
    """Returns the number of variables, limits, constraints, and variable names for Opening"""
    return OpeningAlgorithm()

def closing():
    """Returns the number of variables, limits, constraints, and variable names for Closing"""
    return ClosingAlgorithm()

def morphological_gradient():
    """Returns the number of variables, limits, constraints, and variable names for Morphological Gradient"""
    return MorphologicalGradientAlgorithm()

def top_hat():
    """Returns the number of variables, limits, constraints, and variable names for Top Hat"""
    return TopHatAlgorithm()

def black_hat():
    """Returns the number of variables, limits, constraints, and variable names for Black Hat"""
    return BlackHatAlgorithm()

class ParameterTuningProblem(Problem):
    """Formalize problem as multi-objective optimization"""
    def __init__(self):
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        """Define problem objectives"""
        def minimize_edit_distance(X):
            """Fitness function that minimizes edit distance"""
            results = apply_ocr(X)
            func_results = []
            for text_list in results:
                avg_edit_distance = 0
                for index, text in enumerate(text_list):
                    gt_index = gt_filenames.index(filenames[index])
                    gt_text = gt_files[gt_index]
                    gt_text = ''.join(gt_text)
                    gt_text = clean_text(gt_text)
                    if not text:
                        avg_edit_distance += len(gt_text)
                    else:
                        text = ''.join(text)
                        text = clean_text(text)
                        weighted_levenshtein = WeightedLevenshtein(
                            substitution_cost_fn=substitution_cost,
                            insertion_cost_fn=insertion_cost,
                            deletion_cost_fn=deletion_cost)
                        avg_edit_distance += weighted_levenshtein.distance(text, gt_text)
                avg_edit_distance /= len(text_list)
                func_results.append(avg_edit_distance)
            return func_results

        def maximize_words_matching(X):
            """Fitness function that maximizes the occurrence of words from the ground-truth"""
            results = apply_ocr(X)
            func_results = []
            for text_list in results:
                avg_words_matching = 0
                for index, text in enumerate(text_list):
                    text = ''.join(text)
                    text = clean_text(text)
                    gt_index = gt_filenames.index(filenames[index])
                    gt_text = gt_files[gt_index]
                    gt_text = ''.join(gt_text)
                    gt_text = clean_text(gt_text)
                    gt_vectorizer = CountVectorizer()
                    gt_matrix = gt_vectorizer.fit_transform([gt_text])
                    gt_matrix = gt_matrix.toarray()[0]
                    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None)
                    
                    try:
                        matrix = vectorizer.fit_transform([text])
                        matrix = matrix.toarray()[0]
                        if vectorizer.vocabulary_:
                            success_words = match_dict(gt_vectorizer.vocabulary_, gt_matrix, vectorizer.vocabulary_, matrix)
                            avg_words_matching -= success_words
                    #Exception in case text only contains stop words or does not contain discernible words
                    except ValueError:
                        pass
                avg_words_matching /= len(text_list)
                func_results.append(avg_words_matching)
            return func_results
        
        params = [[X[k]] for k in range(len(X))]

        #Calculate the function values in a parallelized manner and wait until done
        f1 = pool.starmap(minimize_edit_distance, params)
        f2 = pool.starmap(maximize_words_matching, params)
        out["F"] = np.column_stack([f1, f2])

        #Define problem constraints
        if preprocessing_method == "adaptive_threshold":
            out["G"] = abs(X[:,3] % 2 - 1)

        if preprocessing_method == "homogeneous_blur" or preprocessing_method == "median_blur" or preprocessing_method == "gaussian_blur":
            out["G"] = abs(X[:,0] % 2 - 1)

        if preprocessing_method == "homogeneous_blur":
            out["G"] = X[:,0] - X[:,1]

class OddBlockSizeRepair(Repair):
    """Constraint on parameter blockSize (must be an odd number)"""
    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            if x[3] % 2 == 0:
                if x[3] >= xl[3] and x[3] <= xu[3]:
                    if np.random.random() < 0.5:
                        x[3] -= 1
                    else:
                        x[3] += 1
                else:
                    if x[3] >= xu[3]:
                        x[3] -= 1
                    if x[3] <= xl[3]:
                        x[3] += 1
        return pop

class OddKSizeRepair(Repair):
    """Constraint on parameter ksize (must be an odd number)"""
    def _do(self, problem, pop, **kwargs):
        for k in range(len(pop)):
            x = pop[k].X
            if x[0] % 2 == 0:
                if x[0] >= xl[0] and x[0] <= xu[0]:
                    if np.random.random() < 0.5:
                        x[0] -= 1
                    else:
                        x[0] += 1
                else:
                    if x[0] >= xu[0]:
                        x[0] -= 1
                    if x[0] <= xl[0]:
                        x[0] += 1
        return pop

try:
    gt_path = sys.argv[1]
    src_path = sys.argv[2]
    dst_path = sys.argv[3]
    num_iteration = int(sys.argv[4])
    population_size = int(sys.argv[5])
    preprocessing_method = sys.argv[6]
except IndexError:
    print("Invalid command line arguments: 6 arguments are expected. Usage: python genetic_algorithm.py <path to the ground truth path> <path to the sample of digital representations> <dst path for results file> <nummber of iterations> <population size> <image processing method's name>")
    sys.exit()

#Load data
images, filenames = load_images_from_folder(src_path, cv2.IMREAD_GRAYSCALE)
gt_files, gt_filenames = load_files_from_folder(gt_path)
selected_algorithm = call_method(preprocessing_method)
n_var, n_constr, xl, xu, variables = selected_algorithm.get_number_parameters(), selected_algorithm.get_number_constraints(), selected_algorithm.get_lower_bound(), selected_algorithm.get_upper_bound(), selected_algorithm.get_parameters()
custom_config = r'-l por -c classify_enable_learning=0 classify_enable_adaptive_matcher=0'

#Formalize optimization problem with NSGA-II algorithm
problem = ParameterTuningProblem()

#Define NSGA-II algorithm parameterization according to constraints for each image processing algorithm
if preprocessing_method == "adaptive_threshold":
    algorithm = NSGA2(pop_size=population_size,
                repair=OddBlockSizeRepair(),
                sampling=get_sampling("int_random"),
                crossover=get_crossover("int_ux"),
                mutation=get_mutation("int_pm", prob=0.1),
                eliminate_duplicates=True,
                verbose=True)
elif preprocessing_method == "homogeneous_blur" or preprocessing_method == "median_blur" or preprocessing_method == "gaussian_blur":
    algorithm = NSGA2(pop_size=population_size,
                repair=OddKSizeRepair(),
                sampling=get_sampling("int_random"),
                crossover=get_crossover("int_ux"),
                mutation=get_mutation("int_pm", prob=0.1),
                eliminate_duplicates=True,
                verbose=True)
else:
    algorithm = NSGA2(pop_size=population_size,
                sampling=get_sampling("int_random"),
                crossover=get_crossover("int_ux"),
                mutation=get_mutation("int_pm", prob=0.1),
                eliminate_duplicates=True,
                verbose=True)

#For Windows multiprocessing
if __name__ == '__main__':
    #Setup ThreadPool
    pool = ThreadPool(100)

    #Solve optimization problem
    res = minimize(problem, algorithm, ('n_gen', num_iteration), verbose=True)
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    pool.close()

    #Save results to data structure
    data = {}
    for index, value in enumerate(res.X[0]):
        data[variables[index]] = value
    data["edit_distance"]=round(res.F[0][0],2)
    data["bag_of_words"]=-round(res.F[0][1],2)
    data["exec_time"]=round(res.exec_time,2)
    df = pd.DataFrame(data, index=[os.path.basename(os.path.normpath(gt_path))])
    df.index.name="classification"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    dst_file = dst_path + preprocessing_method + ".csv"
    if os.path.isfile(dst_file):
        df.to_csv(dst_file, mode="a", header=False)
    else:
        df.to_csv(dst_file, mode="a")