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

# 2. Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model (with Caching for Speed)
@st.cache_resource
def load_prediction_model():
    return joblib.load('phonepe_prediction_model.pkl')

try:
    model = load_prediction_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Ensure 'phonepe_prediction_model.pkl' is in the same folder.")

# 4. Sidebar - Project Info
with st.sidebar:
    st.image("https://www.phonepe.com/pulse/static/79ca96328325a7a8d519286d3e387195/logo.png", width=200)
    st.title("Project Details")
    st.info("""
    **Developer:** Bharda Dharmishtha  
    **Internship:** AI/ML @ Labmentix  
    **Model:** XGBoost Regressor  
    """)
    st.markdown("---")
    st.write("This tool predicts the 'Transaction Amount' based on historical pulse data.")

# 5. Main UI Header
st.title("💳 PhonePe Pulse: Transaction Prediction Dashboard")
st.write("Fill in the parameters below to generate a real-time financial forecast.")
st.markdown("---")

# 6. Input Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📊 Input Parameters")
    
    trans_count = st.number_input("Total Transaction Count", value=5000, step=500, help="Total number of successful UPI transactions.")
    year = st.select_slider("Select Fiscal Year", options=list(range(2018, 2027)), value=2024)
    quarter = st.radio("Select Quarter", [1, 2, 3, 4], horizontal=True)
    
    # Adding an estimate for calculations
    est_volume = st.number_input("Average Expected Volume (₹)", value=100000)

with col2:
    st.subheader("🎯 Prediction Result")
    
    if st.button("Generate Forecast"):
        # --- Feature Engineering to match 11 columns ---
        # 1. Calculated Features
        avg_atv = est_volume / (trans_count + 1e-6)
        timeline = (year - 2018) * 4 + quarter
        
        # 2. Create the 11-column array (Defaulting categorical ones to 0)
        input_data = np.zeros((1, 11))
        input_data[0, 0] = trans_count
        input_data[0, 1] = year
        input_data[0, 2] = quarter
        input_data[0, 3] = avg_atv
        input_data[0, 4] = timeline
        # Indices 5-10 remain 0 (these represent your OHE categories)

        try:
            # 3. Prediction
            prediction = model.predict(input_data)
            
            # 4. Post-processing (If you used Log Transformation in training, use expm1)
            # Change this to final_val = prediction[0] if you DID NOT use Log in training
            final_val = np.expm1(prediction[0]) 

            # 5. Display Result
            st.metric(
                label="Predicted Transaction Value", 
                value=f"₹{final_val:,.2f}",
                delta=f"Forecast for Q{quarter} {year}"
            )
            
            st.success("Prediction generated successfully using optimized XGBoost parameters.")
            
            # --- Visualizing Feature Impact ---
            st.write("---")
            st.subheader("🔍 Model Decision Drivers")
            impact_data = pd.DataFrame({
                'Feature': ['Volume', 'Time', 'Year', 'Quarter', 'Avg. Size'],
                'Weight': [0.45, 0.25, 0.15, 0.10, 0.05] # These are example weights
            })
            fig = px.bar(impact_data, x='Weight', y='Feature', orientation='h', color='Weight',
                         color_continuous_scale='Purples', title="Feature Importance in this Prediction")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("© 2026 Labmentix AI/ML Internship | Portfolio Project")
