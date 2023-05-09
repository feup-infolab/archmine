import cv2
import numpy as np

class Algorithm:
    """Base class for OpenCV image processing algorithms"""
    def __init__(self, name, parameters, lower_bound, upper_bound, number_constraints):
        self.name=name
        self.parameters=parameters
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.number_constraints=number_constraints
    def get_name(self):
        return self.name
    def get_parameters(self):
        return self.parameters
    def get_number_parameters(self):
        return len(self.parameters)
    def get_lower_bound(self):
        return self.lower_bound
    def get_upper_bound(self):
        return self.upper_bound
    def get_number_constraints(self):
        return self.number_constraints

class Helper:
    """Calls subclasses of OpenCV image processing algorithms"""
    def __init__(self, name):
        self.obj = None
        if name == "homogeneous_blur":
            self.obj = HomogeneousBlurAlgorithm()
        elif name == "gaussian_blur":
            self.obj = GaussianBlurAlgorithm()
        elif name == "median_blur":
            self.obj = MedianBlurAlgorithm()
        elif name == "bilateral_filtering":
            self.obj = BilateralFilteringAlgorithm()
        elif name == "adaptive_threshold":
            self.obj = AdaptiveThresholdAlgorithm()
        elif name == "simple_threshold":
            self.obj = SimpleThresholdAlgorithm()
        elif name == "otsu_threshold":
            self.obj = OtsuThresholdAlgorithm()
        elif name == "triangle_threshold":
            self.obj = TriangleThresholdAlgorithm()
        elif name == "erosion":
            self.obj = ErosionAlgorithm()
        elif name == "dilation":
            self.obj = DilationAlgorithm()
        elif name == "opening":
            self.obj = OpeningAlgorithm()
        elif name == "closing":
            self.obj = ClosingAlgorithm()
        elif name == "morphological_gradient":
            self.obj = MorphologicalGradientAlgorithm()
        elif name == "top_hat":
            self.obj = TopHatAlgorithm()
        elif name == "black_hat":
            self.obj = BlackHatAlgorithm()
        elif name == "baseline":
            self.obj = Baseline()

class HomogeneousBlurAlgorithm(Algorithm):
    def __init__(self):
        """Initializes Homogeneous Blur algorithm with parameter types and boundaries"""
        name="homogeneous_blur"
        parameters=["ksize", "borderType"]
        lower_bound=np.array([1, 0])
        upper_bound=np.array([99, 4])
        super().__init__(name, parameters, lower_bound, upper_bound, 1)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.ksize = values[0]
        if values[1] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[1] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[1] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[1] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[1] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Homogeneous Blur algorithm"""
        return cv2.blur(img, (self.ksize, self.ksize), self.borderType)

class GaussianBlurAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Gaussian Blur algorithm with parameter types and boundaries"""
        name="gaussian_blur"
        parameters=["ksize", "borderType"]
        lower_bound=np.array([3, 0])
        upper_bound=np.array([99, 4])
        super().__init__(name, parameters, lower_bound, upper_bound, 1)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.ksize = values[0]
        if values[1] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[1] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[1] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[1] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[1] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Gaussian Blur algorithm"""
        variables = [img, (self.ksize, self.ksize), 0, 0, self.borderType]
        return cv2.GaussianBlur(*variables)

class MedianBlurAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Median Blur algorithm with parameter types and boundaries"""
        name="median_blur"
        parameters=["ksize"]
        lower_bound=np.array([3])
        upper_bound=np.array([99])
        super().__init__(name, parameters, lower_bound, upper_bound, 1)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.ksize = values[0]

    def get_cv_method(self, img):
        """Returns image after applying Median Blur algorithm"""
        variables = [img, self.ksize]
        return cv2.medianBlur(*variables)

class BilateralFilteringAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Bilateral Filter algorithm with parameter types and boundaries"""
        name="bilateral_filtering"
        parameters=["d", "sigmaColor", "sigmaSpace"]
        lower_bound=np.array([1,0,0])
        upper_bound=np.array([5,255,255])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.d = values[0]
        self.sigmaColor = values[1]
        self.sigmaSpace = values[2]

    def get_cv_method(self, img):
        """Returns image after applying Bilateral Filter algorithm"""
        variables = [img] + list(map(int, [self.d, self.sigmaColor, self.sigmaSpace]))
        return cv2.bilateralFilter(*variables)

class SimpleThresholdAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Simple Thresholding algorithm with parameter types and boundaries"""
        name="simple_threshold"
        parameters=["thresh", "maxValue", "type"]
        lower_bound=np.array([0,1,0])
        upper_bound=np.array([255,255,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.thresh = values[0]
        self.maxValue = values[1]
        if values[2] == 0:
            self.type = cv2.THRESH_BINARY
        elif values[2] == 1:
            self.type = cv2.THRESH_BINARY_INV
        elif values[2] == 2:
            self.type = cv2.THRESH_TRUNC
        elif values[2] == 3:
            self.type = cv2.THRESH_TOZERO
        else:
            self.type = cv2.THRESH_TOZERO_INV

    def get_cv_method(self, img):
        """Returns image after applying Simple Thresholding algorithm"""
        variables = [img] + list(map(int, [self.thresh, self.maxValue, self.type]))
        return cv2.threshold(*variables)[1]

class AdaptiveThresholdAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Adaptive Thresholding algorithm with parameter types and boundaries"""
        name="adaptive_threshold"
        parameters=["maxValue", "adaptiveMethod", "thresholdType", "blockSize", "c"]
        lower_bound=np.array([1,0,0,3,0])
        upper_bound=np.array([255,1,1,99,99])
        super().__init__(name, parameters, lower_bound, upper_bound, 1)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.maxValue = values[0]
        if values[1] == 0:
            self.adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            self.adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if values[2] == 0:
            self.thresholdType = cv2.THRESH_BINARY
        else:
            self.thresholdType = cv2.THRESH_BINARY_INV
        self.blockSize = values[3]
        self.c = values[4]

    def get_cv_method(self, img):
        """Returns image after applying Adaptive Thresholding algorithm"""
        variables = [img] + list(map(int, [self.maxValue, self.adaptiveMethod, self.thresholdType, self.blockSize,self.c]))
        return cv2.adaptiveThreshold(*variables)

class OtsuThresholdAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Otsu Thresholding algorithm with parameter types and boundaries"""
        name="otsu_threshold"
        parameters=["maxValue", "type"]
        lower_bound=np.array([1,0])
        upper_bound=np.array([255,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.maxValue = values[0]
        if values[1] == 0:
            self.type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        elif values[1] == 1:
            self.type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        elif values[1] == 2:
            self.type = cv2.THRESH_TRUNC + cv2.THRESH_OTSU
        elif values[1] == 3:
            self.type = cv2.THRESH_TOZERO + cv2.THRESH_OTSU
        else:
            self.type = cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU

    def get_cv_method(self, img):
        """Returns image after applying Otsu Thresholding algorithm"""
        variables = [img, 0] + list(map(int, [self.maxValue, self.type]))
        return cv2.threshold(*variables)[1]

class TriangleThresholdAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Triangle Thresholding algorithm with parameter types and boundaries"""
        name="triangle_threshold"
        parameters=["maxValue", "type"]
        lower_bound=np.array([1,0])
        upper_bound=np.array([255,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.maxValue = values[0]
        if values[1] == 0:
            self.type = cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        elif values[1] == 1:
            self.type = cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE
        elif values[1] == 2:
            self.type = cv2.THRESH_TRUNC + cv2.THRESH_TRIANGLE
        elif values[1] == 3:
            self.type = cv2.THRESH_TOZERO + cv2.THRESH_TRIANGLE
        else:
            self.type = cv2.THRESH_TOZERO_INV + cv2.THRESH_TRIANGLE

    def get_cv_method(self, img):
        """Returns image after applying Triangle Thresholding algorithm"""
        variables = [img, 0] + list(map(int, [self.maxValue, self.type]))
        return cv2.threshold(*variables)[1]

class ErosionAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Erosion algorithm with parameter types and boundaries"""
        name="erosion"
        parameters=["kernel", "iterations", "borderType"]
        lower_bound=[1,1,0]
        upper_bound=[255,10,4]
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.kernel = values[0]
        self.iterations = values[1]
        if values[2] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[2]  == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[2] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[2]  == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[2]  == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Erosion algorithm"""
        return cv2.erode(img, np.ones((self.kernel, self.kernel), np.uint8), iterations=self.iterations, borderType=self.borderType)

class DilationAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Dilation algorithm with parameter types and boundaries"""
        name="dilation"
        parameters=["kernel", "iterations", "borderType"]
        lower_bound=[1,1,0]
        upper_bound=[255,10,4]
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.kernel = values[0]
        self.iterations = values[1]
        if values[2] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[2] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[2] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[2] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[2] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Dilation algorithm"""
        return cv2.dilate(img, np.ones((self.kernel, self.kernel), np.uint8), iterations=self.iterations, borderType=self.borderType)

class OpeningAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Opening algorithm with parameter types and boundaries"""
        name="opening"
        parameters=["kernel", "iterations", "borderType"]
        lower_bound=np.array([1,1,0])
        upper_bound=np.array([255,10,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.kernel = values[0]
        self.iterations = values[1]
        if values[2] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[2] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[2] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[2] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[2] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Opening algorithm"""
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((self.kernel, self.kernel), np.uint8), iterations=self.iterations, borderType=self.borderType)

class ClosingAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Closing algorithm with parameter types and boundaries"""
        name="closing"
        parameters=["kernel", "iterations", "borderType"]
        lower_bound=np.array([1,1,0])
        upper_bound=np.array([255,10,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.kernel = values[0]
        self.iterations = values[1]
        if values[2] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[2] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[2] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[2] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[2] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Closing algorithm"""
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((self.kernel, self.kernel), np.uint8), iterations=self.iterations, borderType=self.borderType)

class MorphologicalGradientAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Morphological Gradient algorithm with parameter types and boundaries"""
        name="morphological_gradient"
        parameters=["kernel", "iterations", "borderType"]
        lower_bound=np.array([1,1,0])
        upper_bound=np.array([255,10,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.kernel = values[0]
        self.iterations = values[1]
        if values[2] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[2] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[2] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[2] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[2] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Morphological Gradient algorithm"""
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((self.kernel, self.kernel), np.uint8), iterations=self.iterations, borderType=self.borderType)

class TopHatAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Top Hat algorithm with parameter types and boundaries"""
        name="top_hat"
        parameters=["kernel", "iterations", "borderType"]
        lower_bound=np.array([1,1,0])
        upper_bound=np.array([255,10,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.kernel = values[0]
        self.iterations = values[1]
        if values[2] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[2] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[2] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[2] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[2] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Top Hat algorithm"""
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((self.kernel, self.kernel), np.uint8), iterations=self.iterations, borderType=self.borderType)

class BlackHatAlgorithm(Algorithm):
    def __init__(self):
        """Initialize Black Hat algorithm with parameter types and boundaries"""
        name="black_hat"
        parameters=["kernel", "iterations", "borderType"]
        lower_bound=np.array([1,1,0])
        upper_bound=np.array([255,10,4])
        super().__init__(name, parameters, lower_bound, upper_bound, 0)

    def set_parameters(self, values):
        """Assigns parameter values according to input"""
        self.kernel = values[0]
        self.iterations = values[1]
        if values[2] == 0:
            self.borderType = cv2.BORDER_CONSTANT
        elif values[2] == 1:
            self.borderType = cv2.BORDER_REPLICATE
        elif values[2] == 2:
            self.borderType = cv2.BORDER_REFLECT
        elif values[2] == 3:
            self.borderType = cv2.BORDER_REFLECT_101
        elif values[2] == 4:
            self.borderType = cv2.BORDER_ISOLATED

    def get_cv_method(self, img):
        """Returns image after applying Black Hat algorithm"""
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.ones((self.kernel, self.kernel), np.uint8), iterations=self.iterations, borderType=self.borderType)

class Baseline(Algorithm):
    def __init__(self):
        """Initialize baseline scenario where no image processing algorithm is applied to OCR"""
        super().__init__("baseline", [], [], [], 0)