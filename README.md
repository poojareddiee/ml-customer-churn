# Customer Churn Prediction

A machine learning project that predicts customer churn for a telecom company using PyTorch.

## Overview

This project predicts whether a customer is likely to churn based on their tenure, monthly charges, and other account features. The model is trained on the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn).

The project includes:

- Data preprocessing scripts (`churn_preprocessing.py`)
- Model training using PyTorch (`train_pytorch_mlp.py`)
- Saved models (`models/`)
- Streamlit app for interactive predictions (`app.py`)
- Exploratory data analysis notebooks (`notebooks/churn_eda.ipynb`)

## Requirements

- Python 3.12+
- PyTorch
- pandas
- numpy
- scikit-learn
- Streamlit

Install dependencies using:

```bash
pip install -r requirements.txt
