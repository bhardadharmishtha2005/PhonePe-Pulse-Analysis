import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('phonepe_prediction_model.pkl')

st.title("PhonePe Pulse Prediction Dashboard")
st.write("Enter the details to predict Transaction Amount")

# Create input fields based on your features
trans_count = st.number_input("Transaction Count", value=1000)
year = st.slider("Year", 2018, 2026, 2024)
quarter = st.selectbox("Quarter", [1, 2, 3, 4])

# Predict button
if st.button("Predict"):
    # Note: Ensure these features match exactly what your model was trained on
    input_data = np.array([[trans_count, year, quarter]]) 
    prediction = model.predict(input_data)
    st.success(f"Predicted Log Amount: {prediction[0]:.2f}")
