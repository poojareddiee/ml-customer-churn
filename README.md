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
How to Run

Preprocess data:

python3 churn_preprocessing.py


Train the model:

python3 train_pytorch_mlp.py


Run the Streamlit app:

python3 -m streamlit run app.py

Dataset

The dataset used is data/Telco-Customer-Churn.csv (downloaded from Kaggle).

License

This project is for educational purposes.


---

### **3️⃣ Next Steps**  

1. Create `.gitignore` and `README.md` in your project folder.  
2. Stage and commit them:

```bash
git add .gitignore README.md
git commit -m "Add .gitignore and README"
git push
