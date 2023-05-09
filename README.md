# ArchMine

ArchMine is a tool that mines documents with the purpose of automatic population of the ArchOnto ontology.

## Optimization of Image Processing Algorithms for Character Recognition in Cultural Typewritten Documents

An evaluation of the impact of image processing methods and parameter tuning in OCR applied to typewritten cultural heritage documents. The approach uses a multi-objective problem formulation to minimize Levenshtein edit distance and maximize the number of words correctly identified with a non-dominated sorting genetic algorithm (NSGA-II) to tune the methods’ parameters.

The training of image processing algorithms' parameters and evaluation of the OCR optimization process require a dataset to conduct various experiments. We used records from the National Archives of Torre do Tombo (ANTT), a Portuguese central archive with millions of documents that date back to the 9th century, to create a [Cultural Heritage Dataset](https://rdm.inesctec.pt/dataset/cs-2022-004). The dataset has 27,017 one-page digital representations from 8,115 records of two fonds from the 20th century: the General Administration of National Treasury (DGFP) and the National Secretariat of Information (SNI).

To conduct an unbiased evaluation of the OCR optimization process, we used [different samples of the cultural heritage dataset](https://rdm.inesctec.pt/dataset/cs-2022-005) to find the best parameter values for image processing algorithms and to assess the algorithms' performance in the text recognition task. Thus, we have two samples: one to perform the parameter optimization of image processing algorithms (parameterization dataset) and another to evaluate the OCR performance using the algorithms with tuned parameters (evaluation dataset). We applied the NSGA-II algorithm to tune image processing algorithms using the parameterization dataset. We considered two parametrization scenarios: global and by digital representation typology. Global parameter tuning consists of using the overall parametrization sample instead of subsets of the sample by digital representation typology.

The best-performing image processing algorithms with tuned parameters will be used in the pre-processing phase of the OCR task with the population dataset to extract its textual content.