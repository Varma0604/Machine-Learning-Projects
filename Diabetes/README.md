# Diabetes Prediction Using Random Forest Classifier

This project implements a diabetes prediction model using the Pima Indians Diabetes Database. The model is built using a Random Forest Classifier from the Scikit-learn library, and it includes preprocessing steps, model training, evaluation, and visualizations.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Source](#data-source)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Functionality](#functionality)
- [License](#license)

## Introduction

Diabetes is a chronic condition that affects the way your body metabolizes sugar (glucose). Early detection and management are crucial for reducing the risks of severe complications. This project aims to build a predictive model to help identify individuals at risk of diabetes.

## Requirements

To run this project, you'll need the following Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
Installation
Clone this repository or download the code.
Make sure you have Python and the necessary packages installed.
Run the script in your preferred Python environment.
Usage
Execute the script to load the dataset, preprocess the data, and train the model.
The model will automatically evaluate its performance on a test set.
Example usage for predicting diabetes for a new individual is included at the end of the script.
Data Source
The dataset used in this project is the Pima Indians Diabetes Database, which can be found here.

The dataset consists of the following features:

pregnancies: Number of pregnancies
glucose: Glucose level
blood_pressure: Blood pressure level
skin_thickness: Skin thickness measurement
insulin: Insulin level
bmi: Body mass index
diabetes_pedigree_function: Diabetes pedigree function
age: Age of the individual
outcome: Class variable (0: non-diabetic, 1: diabetic)
Model Evaluation
After training, the model evaluates its performance using:

Classification Report
Confusion Matrix
Accuracy Score
The results provide insights into the model's predictive power.

Visualizations
The project generates visualizations for:

Feature Importance: Displays the importance of each feature in predicting diabetes.
Confusion Matrix: Visualizes the model's performance in classifying diabetes outcomes.
Both visualizations are saved as PNG files: feature_importance.png and confusion_matrix.png.

Functionality
A function predict_diabetes(new_data) is provided to make predictions for new input data. An example of how to use this function is included in the script.

Example Prediction
python
Copy code
new_person = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
prediction, probability = predict_diabetes(new_person)
print(f"Prediction for new person: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
print(f"Probability of having diabetes: {probability:.2f}")
License
This project is licensed under the MIT License. Feel free to use and modify it as needed.