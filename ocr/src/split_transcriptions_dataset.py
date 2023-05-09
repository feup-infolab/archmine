import sys
import os
import random
import shutil

"""Split dataset of transcriptions in two: parameterization and evaluation datasets"""

try:
    images_sample_folder = sys.argv[1]
    transcriptions_sample_folder = sys.argv[2]
    images_parametrization_folder = sys.argv[3]
    images_evaluation_folder = sys.argv[4]
    transcriptions_parametrization_folder = sys.argv[5]
    transcriptions_evaluation_folder = sys.argv[6]
except IndexError:
    print("Invalid command line arguments: 6 arguments are expected. Usage: python split_transcriptions_dataset.py <folder with digital representations> <folder with transcriptions> <dst folder of parametrization sample of digital representations> <dst folder of evaluation sample of digital representations> <dst folder of parametrization sample of transcriptions> <dst folder of evaluation sample of transcriptions>")
    sys.exit()

filenames=os.listdir(transcriptions_sample_folder)
sample_size = int(len(filenames)/2)
files_per_register = {}
for filename in filenames:
    register = filename.split('_')[0]
    if register not in files_per_register:
        files_per_register[register] = [filename]
    else:
        files_per_register[register].append(filename)

parametrization_sample=[]
evaluation_sample=[]

while len(parametrization_sample) != sample_size and len(evaluation_sample) != sample_size:
    if len(list(files_per_register)) >= 2:
        registers=random.sample(list(files_per_register),2)
        param_chosen_register=registers[0]
        eval_chosen_register=registers[1]

        param_chosen_files=files_per_register[param_chosen_register]
        eval_chosen_files=files_per_register[eval_chosen_register]

        if len(parametrization_sample) + len(param_chosen_files) <= sample_size:
            if param_chosen_register in files_per_register:
                parametrization_sample.extend(param_chosen_files)
                del files_per_register[param_chosen_register]
        if len(evaluation_sample) + len(eval_chosen_files) <= sample_size:
            if eval_chosen_register in files_per_register:
                evaluation_sample.extend(eval_chosen_files)
                del files_per_register[eval_chosen_register]
    else:
        chosen_register=list(files_per_register)[0]
        param_chosen_files=random.sample(files_per_register[chosen_register],sample_size-len(parametrization_sample))
        eval_chosen_files=[filename for filename in files_per_register[chosen_register] if not filename in param_chosen_files]

        if chosen_register in files_per_register:
            parametrization_sample.extend(param_chosen_files)
            evaluation_sample.extend(eval_chosen_files)

if len(parametrization_sample) == sample_size:
    evaluation_sample.extend([filename for filename in filenames if not filename in parametrization_sample])

if len(evaluation_sample) == sample_size:
    parametrization_sample.extend([filename for filename in filenames if not filename in evaluation_sample])

if not os.path.exists(transcriptions_parametrization_folder):
    os.makedirs(transcriptions_parametrization_folder)

if not os.path.exists(transcriptions_evaluation_folder):
    os.makedirs(transcriptions_evaluation_folder)

digital_representations=os.listdir(images_sample_folder)

if not os.path.exists(images_parametrization_folder):
    os.makedirs(images_parametrization_folder)

if not os.path.exists(images_evaluation_folder):
    os.makedirs(images_evaluation_folder)

for file in parametrization_sample:
    shutil.copy(os.path.join(transcriptions_sample_folder, file), transcriptions_parametrization_folder)
    for image in digital_representations:
        if os.path.splitext(file)[0] in image:
            shutil.copy(os.path.join(images_sample_folder, image), images_parametrization_folder)

for file in evaluation_sample:
    shutil.copy(os.path.join(transcriptions_sample_folder, file), transcriptions_evaluation_folder)
    for image in digital_representations:
        if os.path.splitext(file)[0] in image:
            shutil.copy(os.path.join(images_sample_folder, image), images_evaluation_folder)

print("Saved files.")