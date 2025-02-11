# **Breast Cancer Detection**

## Overview:
This project implements a breast cancer detection system using machine learning. It utilizes logistic regression to classify tumors as malignant or benign based on medical features. The dataset used is breast-cancer.csv.

## Features:

Data Preprocessing: Cleans and prepares data for training.

Exploratory Data Analysis (EDA): Visualizes feature distributions and correlations.

Feature Selection: Identifies the most relevant features.

Model Training: Implements Logistic Regression for classification.

Model Evaluation: Computes accuracy and makes predictions.

## Dataset:

The dataset contains various features extracted from cell nuclei in breast cancer biopsies. The target variable is diagnosis, which is converted to binary (1 = Malignant, 0 = Benign).

## Installation:

Clone the repository:

git clone https://github.com/RutviSindha/breast-cancer-detection.git

Navigate to the project directory:

cd breast-cancer-detection

Install dependencies:

pip install -r requirements.txt

## Usage

Ensure you have the breast-cancer.csv dataset in the project directory.

Run the script:

python Breast-Cancer-Detection.py

The model will train and output accuracy and predictions.

## Dependencies

Python 3.x

pandas

numpy

seaborn

matplotlib

plotly

scikit-learn

## Model Evaluation

The model uses Logistic Regression for classification.

It achieves an accuracy score that is displayed upon execution.

Feature selection improves model performance by reducing irrelevant data.

## Results

The model achieves an accuracy of **96.49%** on test data. It evaluates accuracy using test data and provides predictions for individual samples.
