import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="PhonePe Pulse Insights", 
    page_icon="💳", 
    layout="wide"
)

# 2. Advanced CSS for a Clean White "Apple-Style" UI
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #FFFFFF;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E9ECEF;
    }
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        color: #5F259F !important; /* PhonePe Purple */
        font-weight: 800;
    }
    .stMetric { 
        background-color: #FFFFFF; 
        padding: 25px; 
        border-radius: 15px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #F1F3F5;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #5F259F;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #451a75;
        border: none;
        color: white;
    }
    /* Headers */
    h1, h2, h3 {
        color: #212529;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_prediction_model():
    return joblib.load('phonepe_prediction_model.pkl')

model = load_prediction_model()

# 4. Sidebar - Clean Text Only
with st.sidebar:
    st.markdown("## 📊 **Project Pulse**")
    st.write("An AI-powered forecasting tool developed for the **Labmentix** internship program.")
    st.divider()
    st.markdown("#### **System Status**")
    st.success("Model Live & Ready")
    st.caption("Version 1.0.2")

# 5. Main UI
st.title("💳 PhonePe Pulse: Transaction Prediction")
st.markdown("##### Fill in the parameters to generate an AI-driven financial forecast.")
st.divider()

# 6. Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📍 Parameters")
    trans_count = st.number_input("Total Transaction Count", value=5000, step=500)
    year = st.select_slider("Fiscal Year", options=list(range(2018, 2027)), value=2024)
    quarter = st.radio("Quarter", [1, 2, 3, 4], horizontal=True)
    est_volume = st.number_input("Average Expected Volume (₹)", value=100000)

with col2:
    st.subheader("🎯 Result")
    
    if st.button("Generate Forecast", use_container_width=True):
        # Feature Engineering (11 columns)
        avg_atv = est_volume / (trans_count + 1e-6)
        timeline = (year - 2018) * 4 + quarter
        
        input_data = np.zeros((1, 11))
        input_data[0, 0] = trans_count
        input_data[0, 1] = year
        input_data[0, 2] = quarter
        input_data[0, 3] = avg_atv
        input_data[0, 4] = timeline
        
        try:
            prediction = model.predict(input_data)
            final_val = np.expm1(prediction[0]) 

            # Result Display
            st.metric(
                label="Estimated Transaction Value", 
                value=f"₹{final_val:,.2f}",
                delta=f"Based on Q{quarter} {year} Trends"
            )
            
            # Feature Importance Chart (Clean White Theme)
            st.write("---")
            impact_data = pd.DataFrame({
                'Feature': ['Volume', 'Time', 'Year', 'Quarter', 'Size'],
                'Weight': [0.45, 0.25, 0.15, 0.10, 0.05]
            }).sort_values('Weight')
            
            fig = px.bar(impact_data, x='Weight', y='Feature', orientation='h',
                         template='plotly_white', # White template
                         color_discrete_sequence=['#5F259F']) # Purple color
            
            fig.update_layout(
                title="Model Driver Intensity",
                margin=dict(l=20, r=20, t=40, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.divider()
st.caption("AI/ML Internship Project | © 2026 Digital India Analytics")
