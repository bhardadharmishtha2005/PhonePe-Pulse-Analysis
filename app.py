import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(page_title="PhonePe Pulse Insights", page_icon="💳", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #673ab7; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('phonepe_prediction_model.pkl')

model = load_model()

# Header Section
st.title("💳 PhonePe Pulse: Transaction Prediction")
st.markdown("---")

# Layout with Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Input Parameters")
    trans_count = st.number_input("Transaction Count (Total)", value=1000, step=100)
    year = st.select_slider("Select Fiscal Year", options=list(range(2018, 2027)), value=2024)
    quarter = st.radio("Select Quarter", [1, 2, 3, 4], horizontal=True)
    amount_guess = st.number_input("Expected Regional Volume (₹)", value=50000)

with col2:
    st.subheader("💡 Analysis & Prediction")
    st.info("The model uses XGBoost to calculate the likely transaction value based on historical regional trends.")
    
    if st.button("Generate Prediction"):
        # Engineering features to match 11-column shape
        avg_atv = amount_guess / (trans_count + 1e-6)
        timeline = (year - 2018) * 4 + quarter
        
        # Creating input for model (adjust order based on your X_train_final.columns)
        input_data = np.zeros((1, 11))
        input_data[0, 0] = trans_count
        input_data[0, 1] = year
        input_data[0, 2] = quarter
        input_data[0, 3] = avg_atv
        input_data[0, 4] = timeline
        
        try:
            prediction = model.predict(input_data)
            final_val = np.expm1(prediction[0])
            
            st.metric(label="Predicted Transaction Value", value=f"₹{final_val:,.2f}")
            st.success("Prediction calculated successfully!")
            
            # Additional logic for mentor-friendly "impact"
            st.write(f"**Insight:** At this volume, the average ticket size is ₹{avg_atv:,.2f} per user.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
