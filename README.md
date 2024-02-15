# Apple-Defects-Classifier

This repository contains a deep learning-based classifier for apple defects. The classifier can identify and classify various common types of defects in apples, such as bruises, spots, and rot.

# Directory Structure
**acquisition**: Contains code for initializing cameras and capturing images of apples.
**architectures**: Contains test files for the model and code for generating the confusion matrix. It also includes two subdirectories named feedforward and siamese, which contain code for models trained using those types of neural networks.
**preprocessing**: Contains code for defect segmentation and data augmentation.

# Requirements
Python 3.x
TensorFlow
OpenCV
Matplotlib
Numpy
Other requirements specified in requirements.txt

# Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/your_username/Apple-Defect-Classifier.git
```
Navigate to the repository directory:
```bash
cd Apple-Defect-Classifier
```
Install dependencies using pip:
```bash
pip install -r requirements.txt
```
# Usage
To use the apple defect classifier, follow these steps:

Ensure you have your apple images in a suitable directory structure within the datasets folder.
Run the preprocessing script to prepare the training, validation, and test data.
Train the model using the provided training script.
Once trained, you can use the classifier to predict defects in new apple images using the prediction script.

# Models

[View models](https://espolec-my.sharepoint.com/:f:/g/personal/omcoello_espol_edu_ec/EgseUQqxOzBOliO_ySzZZ-EBpUdKWs6TDicWPVONfvMEMg?e=3hCJEq) trained by epochs with h5 format and also with keras format. They are classified by spectrum and model used.
