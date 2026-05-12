# Customer Churn Prediction using Explainable Machine Learning

## Module
CSC-44112 Advanced Applications of AI and Machine Learning

## Assessment
Assessment Part 2 – Technical Data Science Report

## Project Title
Predicting Customer Churn Using Explainable Machine Learning Techniques

## Overview
This project builds an end-to-end customer churn prediction pipeline using the IBM Telco Customer Churn dataset.

The project includes:

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Feature encoding and scaling
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network
- Model evaluation
- XGBoost feature importance explainability
- Report-ready visual outputs

## Dataset
Download the dataset here:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place this file in the project folder:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is not included in this repository to avoid redistribution issues.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python churn_prediction_project.py
```

## Outputs
The script creates an `outputs/` folder containing:

- EDA figures
- model comparison table
- ROC curve
- confusion matrix
- XGBoost feature importance
- neural network learning curve

## Author
Student project for CSC-44112.
