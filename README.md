# Telco Customer Churn Prediction

This project aims to predict customer churn (whether a customer will leave the service) using a dataset from a fictional telecom company. The project is built using Streamlit for the web interface, TensorFlow/Keras for building a neural network, and Scikit-learn for data preprocessing and evaluation.

## Table of Contents
*Project Overview
Features
Installation
Usage
Data Description
Model Description
Evaluation*

## Project Overview

**This project predicts whether customers will churn based on various factors like monthly charges, contract type, and tenure. It utilizes the following:**

*Neural Network for classification
SMOTE for handling class imbalance
K-Fold Cross-Validation for better model generalization
Data Visualizations to understand relationships between features*

**The project provides users with:**
Data exploration tools for visualizing churn patterns
Training a predictive model
Predicting churn for new customers based on their characteristics

## Features
**1.Data Visualization**: *Explore churn data with various plots, including distribution and correlation analysis.*
**2.Train Model**: *Train a Sequential Neural Network to predict churn with options for:*
**- Regular model training 
  - K-Fold Cross-Validation**
**3.Predict Churn**: *Predict whether a new customer will churn based on their input features.*
**4.Model** *Evaluation: Visualize model performance through accuracy, loss, and classification reports.*

## Installation
**Follow these steps to set up the project locally:**

**Clone the repository:**

`git clone https://github.com/your-repo/telco-churn-prediction.git`


**Navigate to the project directory:**
    
`cd telco-churn-prediction`

**Install the required packages using pip:**

`pip install -r requirements.txt`
**Required Libraries:
Streamlit
Pandas
Seaborn
Matplotlib
Scikit-learn
Imbalanced-learn
TensorFlow/Keras**

**Run the Streamlit app:**

`streamlit run app.py`


## Usage

Once the app is running, you will have access to three main functionalities:

**1.Data Visualization:**

*Explore the dataset through various plots such as churn count, churn by contract type, and churn rate by categorical features.
View numeric distributions and correlations between features.*

**2.Train Model:**

*Train a model using the Sequential Neural Network. You can choose either standard model training or use K-Fold Cross-Validation.
Visualize the accuracy and loss history during training.*

**3.Predict Churn:**

*Input customer details into the form and predict whether the customer is likely to churn.*


## Data Description

**The dataset used is the Telco Customer Churn Dataset, which includes customer attributes such as:**

*-gender

-SeniorCitizen

-Partner

-Dependents

-tenure

-PhoneService, MultipleLines

-InternetService, OnlineSecurity, OnlineBackup

-DeviceProtection, TechSupport, StreamingTV, StreamingMovies

-Contract, PaperlessBilling, PaymentMethod

-MonthlyCharges, TotalCharges

-Churn (Target Variable)*

## The dataset is preprocessed with:

*1.Handling missing values in TotalCharges
2.One-hot encoding for categorical features
3.Standard scaling for numerical features*

## Model Description
**The Sequential Neural Network consists of:**

*Input layer with features from the dataset
Multiple hidden layers with ReLU activation, batch normalization, and dropout for regularization
An output layer with a sigmoid activation for binary classification
Loss function: Binary Cross-Entropy
Optimizer: Adam*


**SMOTE** (Synthetic Minority Oversampling Technique) is applied to handle the class imbalance between churned and non-churned customers.

## Evaluation
**The model is evaluated using:**
*Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1-score)*

Visualizations of training accuracy, validation accuracy, training loss, and validation loss are provided. You can also perform K-Fold Cross-Validation to assess the model's performance across different splits.*