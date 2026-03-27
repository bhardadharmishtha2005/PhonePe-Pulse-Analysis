import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load('phonepe_prediction_model.pkl')

st.title("PhonePe Pulse Prediction Dashboard")

# 1. User Inputs
trans_count = st.number_input("Transaction Count", value=1000)
year = st.slider("Year", 2018, 2026, 2024)
quarter = st.selectbox("Quarter", [1, 2, 3, 4])
# Adding a mock input for Amount to calculate Avg_Transaction_Value if needed
amount_guess = st.number_input("Typical Amount for this region", value=50000)

if st.button("Predict"):
    # 2. FEATURE ENGINEERING (This must match your training!)
    avg_atv = amount_guess / (trans_count + 1e-6)
    timeline = (year - 2018) * 4 + quarter
    
    # 3. Create the input array with ALL features in the correct order
    # Add or remove variables here based on what X_train_final.columns shows
    features = np.array([[trans_count, year, quarter, avg_atv, timeline]])
    
    try:
        prediction = model.predict(features)
        # Since we used Log transformation, we convert it back using expm1
        final_val = np.expm1(prediction[0])
        st.success(f"Predicted Transaction Amount: ₹{final_val:,.2f}")
    except ValueError as e:
        st.error(f"Feature Mismatch: The model expects different inputs. {e}")
