# Optimization of Image Processing Algorithms for Character Recognition in Cultural Typewritten Documents

## Dependencies

* [Tesseract](https://tesseract-ocr.github.io/tessdoc/Home.html) (4.0 or above)
* [Portuguese trained data file for Tesseract](https://github.com/tesseract-ocr/tessdata_best/blob/main/por.traineddata) (add to tessdata folder)

## Installation

The `requirements.txt` file lists all Python libraries that the OCR tool depends on and will be installed using:

```
python install -r requirements.txt
```

The Tesseracted executable file must be added to your PATH in order to use the PyTesseract wrapper.

## Run

To download the dataset and apply OCR to the digital representations with the selected image processing algorithms per document type, run the command:

```
./run-ocr.sh
```

To run the parameterization of different image processing algorithms, execute the command:

```
./run-parameterization.sh
```

To run the evaluation of the optimization experiment, run the command:

```
./run-evaluation.sh
```