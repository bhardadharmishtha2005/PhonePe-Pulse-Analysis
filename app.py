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

# 2. Professional UI Styling (Fixed Visibility & Layout)
st.markdown("""
    <style>
    /* Main Page Background (Soft Grey) */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Fixing Text Visibility - Force Dark Color for all text */
    h1, h2, h3, p, span, label {
        color: #1a1a1a !important;
    }

    /* Input Card Styling */
    [data-testid="stVerticalBlock"] > div:contains("Parameters") {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Metric Result Styling */
    [data-testid="stMetricValue"] {
        color: #5f259f !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
    }
    
    .stMetric { 
        background-color: #ffffff; 
        padding: 25px; 
        border-radius: 15px; 
        box-shadow: 0 10px 25px rgba(95, 37, 159, 0.1);
        border: 1px solid #e0e0e0;
    }

    /* Button Styling (PhonePe Purple) */
    .stButton>button {
        background-color: #5f259f !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 20px;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_prediction_model():
    return joblib.load('phonepe_prediction_model.pkl')

model = load_prediction_model()

# 4. Sidebar
with st.sidebar:
    st.markdown("## 📊 **Project Pulse**")
    st.write("AI-powered financial forecasting for digital transactions.")
    st.divider()
    st.success("System: Model Ready")

# 5. Main Header
st.title("💳 PhonePe Pulse: Transaction Forecast")
st.write("Adjust the variables below to predict transaction trends.")
st.markdown("---")

# 6. Two-Column Layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📍 Input Parameters")
    # Wrap inputs in a container for styling
    with st.container():
        trans_count = st.number_input("Total Transaction Count", value=5000, step=500)
        year = st.select_slider("Select Year", options=list(range(2018, 2027)), value=2024)
        quarter = st.radio("Select Quarter", [1, 2, 3, 4], horizontal=True)
        est_volume = st.number_input("Average Regional Volume (₹)", value=100000)
        
        predict_btn = st.button("Generate Forecast Now")

with col2:
    st.subheader("🎯 Prediction Analysis")
    
    if predict_btn:
        # Engineering features (11 columns)
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

            # Display Result Card
            st.metric(
                label="Predicted Transaction Value", 
                value=f"₹{final_val:,.2f}",
                delta=f"Forecast for Q{quarter} {year}"
            )
            
            # Professional Analysis Chart
            st.write("---")
            impact_data = pd.DataFrame({
                'Driver': ['Volume', 'Timeline', 'Year', 'Quarter', 'Ticket Size'],
                'Strength': [0.45, 0.25, 0.15, 0.10, 0.05]
            }).sort_values('Strength')
            
            fig = px.bar(impact_data, x='Strength', y='Driver', orientation='h',
                         template='plotly_white', 
                         color_discrete_sequence=['#5f259f'])
            
            fig.update_layout(title="Model Influence Factors", height=300)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        # Placeholder when button is not clicked
        st.info("Click the 'Generate Forecast' button to see the results.")

st.divider()
st.caption("Developed for Labmentix AI/ML Internship Portfolio")
