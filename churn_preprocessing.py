# churn_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Drop customerID
df = df.drop('customerID', axis=1)

# Convert 'TotalCharges' to numeric and fill missing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Feature engineering
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,72], labels=False)
df['MonthlyCharges_x_Tenure'] = df['MonthlyCharges'] * df['tenure']

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Data preprocessing complete!")
