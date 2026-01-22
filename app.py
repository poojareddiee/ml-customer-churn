import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from train_pytorch_mlp import ChurnMLP  # import the model class

# Define numeric features
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# Load model
input_dim = len(numeric_features)
model = ChurnMLP(input_dim)
model.load_state_dict(torch.load("models/churn_model.pth"))
model.eval()

st.title("Customer Churn Prediction")

# Input fields
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0)

if st.button("Predict Churn"):
    # Create input tensor
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    # For simplicity, we fit scaler on single row
    X_input = np.array([[tenure, monthly_charges, total_charges]], dtype=float)
    X_input_scaled = scaler.fit_transform(X_input)
    X_input_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)
    
    # Get prediction
    with torch.no_grad():
        logits = model(X_input_tensor)
        prob = torch.sigmoid(logits).item()
    
    st.write(f"Predicted Churn Probability: {prob:.2f}")
    if prob > 0.5:
        st.warning("Customer is likely to churn!")
    else:
        st.success("Customer is unlikely to churn.")
