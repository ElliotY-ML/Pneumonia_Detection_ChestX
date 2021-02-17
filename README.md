# Pneumonia Detection From X-Rays
This repository contains a completed cap-stone project for Udacity's "Applying AI to 2D Medical Imaging Data" course, 
part of the AI for Healthcare Nanodegree program.  It has been reviewed by Udacity instructors and met project specifications.

**Introduction**  
Advancements in machine learning and computer vision allow new opportunities to create software to assist medical
physicians.  Assistive software can improve patient prioritization,
reduce physicians' efforts to examine medical images, or introduce consistent measurements of anatomy from medical images.
In this project, computer vision with a convolutional neural network (CNN) model is trained to predict the presence 
of pneumonia from chest X-Ray images.  The intention of this software is to pre-screen given chest X-Ray images prior to radiologists' review
and classify the X-Ray for the presence or absence of pneumonia.  

This project is broken is three Jupyter Notebooks:  
- 1_EDA (Exploratory Data Analysis): Dataset metadata analysis and X-ray image pixel-level analysis. 
- 2_Build_and_Train_Model: Image pre-processing with Keras ImageDataGenerator, split dataset using Scikit-Learn, build & train a Keras Sequential model, 
and convert probabilistic outputs to binary predictions.  
- 3_Inference:  DICOM pixel data extraction, normalize & standardize pixel data, and apply trained model to make predictions.


**Fine Tuning Convolutional Neural Network VGG16 for Pneumonia Detection from X-Rays**  
This project's model was created by fine-tuning ImageNet's VGG16 CNN model with chest X-Ray images.  
To fine-tune the VGG16 model, a new Keras Sequential model was created by taking VGG16 model layers 
and freezing their ImageNet-trained weights.  Subsequent Dense and Dropout layers were added, which will have their weights trained 
for classifying chest X-Ray images for pneumonia.
Model predictions initially return as probabilities between 0 and 1.  These probabilistic results were compared 
against ground truth labels.  A threshold analysis was completed to select the boundary at which 
probalistic results are converted into binary results of either pneumonia presence or absence.
 
The paper of Pranav Rajpurkar et al. (2017), "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning", 
provides a reference to compare against.  This paper established F1-scores as the performance metric to compare radiologists' and algorithms' 
performance in identifying pneumonia in a subset of 420 images from the ChestX-ray14 dataset (Wang et al., 2017). 
F1-scores are the harmonic average of the precision and recall of a model's predictions against ground truth labels.
The CheXNet algorithm achieved an F1 score of 0.435, while a panel of four independent Radiologists averaged an F1 score of 0.387. 
This project's final F1 score is 0.36, which is similar in performance to the panel of Radiologist. 

- For further information about the model architecture, please read the "Algorithm Design and Function" section of 
the [`FDA_Preparation.md`](https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX/blob/master/FDA_Preparation.md).
- Please read [`2_Build_and_Train_Model.ipynb`](https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX/blob/master/2_Build_and_Train_Model.ipynb) for full details of model training and threhold selection.

**References**
[1]  Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, Matthew P. Lungren, Andrew Y. Ng, "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning,"  arXiv:1711.05225, Dec 2017. [Link](https://arxiv.org/abs/1711.05225)   
[2]  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017 


**Making Predictions**  
The [`3_Inference Jupyter Notebook`](https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX/blob/master/3_Inference.ipynb)
contains the functions to load DICOM files, pre-process DICOM image, 
load the model built in 2_Build_and_Train_Model, and predict the presence of pneumonia from the DICOM image.

Inputs:
- .dcm DICOM medical imaging file, contains metadata and a medical image

Output:
- DICOM image is displayed with a prediction of whether the patient is Positive or Negative for Pneumonia

The following steps should be performed to analyze a chest X-Ray DICOM file:
1.  Load DICOM file with `check_dicom(filename)` function.  It's output is the DICOM pixel_array or 
an error message if the DICOM file is not a Chest X-Ray.    
2.  Pre-process the loaded DICOM image with `preprocess_image(img=pixel_array, img_mean=0, img_std=1, img_size=(1,224,224,3))` function.
3.  Load trained model with `load_model(model_path, weight_path)`.
4.  Make prediction with `predict_image(model, img, thresh=0.245)`.


### Dataset
The ChestX-ray14 dataset was  curated by Wang et al. and was released by NIH Clinical Center.
It is comprised of 112,120 X-Ray images with disease labels from 30,805 unique patients. 
The disease labels for each image were created using Natural Language Processing (NLP) to process 
associated radiological reports for fourteen common pathologies. The estimated accuracy of the NLP labeling accuracy is estimated to be >90%.

**References**
[1]  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017 

## Getting Started

1. Set up your Anaconda environment.  
2. Clone `https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX` GitHub repo to your local machine.
3. Open `1_EDA.ipynb` with Jupyter Notebook for exploratory data analysis.
4. Open `2_Build_and_Train_Model.ipynb` with Jupyter Notebook for image pre-processing with keras ImageDataGenerator, 
ImageNet VGG16 CNN model fine-tuning, and threshold analysis.
5. Open `3_Inference.ipynb` with Jupyter Notebook for inference with a DICOM file.
6. Complete Project Discussion can be found in `FDA_Preparation.md`

### Dependencies
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

**Create local environment**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
**CHANGE**
```
git clone https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX.git
cd Pneumonia_Detection_ChestX
```

2. Create (and activate) a new environment, named `udacity-ehr-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n udacity-ehr-env python=3.7
	source activate udacity-ehr-env
	```
	- __Windows__: 
	```
	conda create --name udacity-ehr-env python=3.7
	activate udacity-ehr-env
	```
	
	At this point your command line should look something like: `(udacity-ehr-env) <User>:USER_DIR <user>$`. The `(udacity-ehr-env)` indicates that your environment has been activated, and you can proceed with further package installations.



6. Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
 
```
pip install -r pkgs.txt
```


## Project Instructions
please read Udacity's original project instructions in the `Project_Overview.md` file.

**Project Overview**

   1. Exploratory Data Analysis
   2. Building and Training Your Model
   3. Clinical Workflow Integration
   4. FDA Preparation


## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md)
