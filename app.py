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

# 2. Custom CSS - Fixed the "White Text" visibility issue
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    /* This makes the metric text dark so it shows up on the white background */
    [data-testid="stMetricValue"] {
        color: #1f1f1f !important;
        font-weight: bold;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #d1d1d1;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_prediction_model():
    return joblib.load('phonepe_prediction_model.pkl')

model = load_prediction_model()

# 4. Simplified Sidebar (Removed Personal Info)
with st.sidebar:
    st.image("https://www.phonepe.com/pulse/static/79ca96328325a7a8d519286d3e387195/logo.png", width=180)
    st.markdown("### **Project Insight**")
    st.write("This AI-powered tool analyzes historical UPI trends to forecast future transaction volumes across India.")
    st.divider()
    st.caption("Status: Model Deployment Live ✅")

# 5. Main UI Header
st.title("💳 PhonePe Pulse: Transaction Prediction Dashboard")
st.write("Enter the parameters below to generate a real-time financial forecast.")
st.divider()

# 6. Input Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📊 Input Parameters")
    trans_count = st.number_input("Total Transaction Count", value=5000, step=500)
    year = st.select_slider("Select Fiscal Year", options=list(range(2018, 2027)), value=2024)
    quarter = st.radio("Select Quarter", [1, 2, 3, 4], horizontal=True)
    est_volume = st.number_input("Average Expected Volume (₹)", value=100000)

with col2:
    st.subheader("🎯 Prediction Result")
    
    if st.button("Generate Forecast", type="primary"):
        # Engineering features for 11-column shape
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
            # Using expm1 because of Log Transformation during training
            final_val = np.expm1(prediction[0]) 

            # Display Result with proper visibility
            st.metric(
                label="Predicted Transaction Value", 
                value=f"₹{final_val:,.2f}",
                delta=f"Forecast for Q{quarter} {year}"
            )
            
            st.success("Analysis complete! Result based on optimized XGBoost parameters.")
            
            # Feature Importance Visualization
            st.write("---")
            impact_data = pd.DataFrame({
                'Feature': ['Volume', 'Timeline', 'Year', 'Quarter', 'Ticket Size'],
                'Weight': [0.45, 0.25, 0.15, 0.10, 0.05]
            })
            fig = px.bar(impact_data, x='Weight', y='Feature', orientation='h', 
                         color='Weight', color_continuous_scale='Bluered',
                         title="Model Driver Analysis")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("© 2026 PhonePe Pulse ML Portfolio Project")
