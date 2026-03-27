import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model = joblib.load('phonepe_prediction_model.pkl')

# Renamed Title to match GitHub Repo
st.title("PhonePe Pulse Analysis Dashboard")
st.write("Predicting Transaction Amounts using XGBoost")

# 1. User Inputs (The main drivers)
trans_count = st.number_input("Transaction Count", value=1000)
year = st.slider("Year", 2018, 2026, 2024)
quarter = st.selectbox("Quarter", [1, 2, 3, 4])
amount_guess = st.number_input("Typical Amount for this region", value=50000)

if st.button("Predict"):
    # 2. Calculating secondary features
    avg_atv = amount_guess / (trans_count + 1e-6)
    timeline = (year - 2018) * 4 + quarter
    
    # 3. Handling the 11 Features
    # We create a list of 11 zeros first
    final_features = np.zeros((1, 11))
    
    # Fill the first 5 with our inputs (Check your Colab to ensure the order is correct!)
    final_features[0, 0] = trans_count
    final_features[0, 1] = year
    final_features[0, 2] = quarter
    final_features[0, 3] = avg_atv
    final_features[0, 4] = timeline
    # The remaining 6 (indices 5 to 10) will stay as 0 (representing 'Other' categories)
    
    try:
        prediction = model.predict(final_features)
        # Convert back from Log if you used log transformation earlier
        final_val = np.expm1(prediction[0])
        st.success(f"Predicted Transaction Amount: ₹{final_val:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
