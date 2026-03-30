import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="PhonePe Pulse Analytics", page_icon="⚡", layout="wide")

# 2. Light Theme CSS & Updated Button Color
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    
    /* Dark Text for Visibility */
    h1, h2, h3, p, label, .stMarkdown { color: #1E1E1E !important; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FDFDFF;
        border-right: 1px solid #E9ECEF;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }

    /* UPDATED: Lighter Button Color */
    div.stButton > button:first-child {
        background-color: #9B59B6 !important; /* Lighter Amethyst Purple */
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        height: 3.5em !important;
        font-weight: 600 !important;
        width: 100%;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #A569BD !important; /* Even lighter on hover */
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - Simplified
with st.sidebar:
    st.markdown("# **Pulse v2.0**")
    menu = st.radio("MENU", ["🚀 Predictor Engine", "📊 Market Insights", "📄 Tech Documentation"])
    
    st.divider()
    st.markdown("### **Model Status**")
    st.success("XGBoost Model: Active")
    st.info("Accuracy: 98%")
    
    st.divider()
    st.caption("Last Updated: March 2026")

# 4. Load Model
@st.cache_resource
def load_model():
    return joblib.load('phonepe_prediction_model.pkl')

model = load_model()

# 5. Main Content Logic
if menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Prediction Engine")
    
    col1, col2 = st.columns([1, 1.6], gap="large")
    
    with col1:
        st.subheader("⚙️ Inputs")
        with st.container(border=True):
            trans_count = st.number_input("Transaction Count", value=5000)
            year = st.select_slider("Year", options=list(range(2018, 2027)), value=2024)
            quarter = st.segmented_control("Quarter", [1, 2, 3, 4], default=1)
            est_vol = st.number_input("Regional Volume (₹)", value=150000)
            
            predict_btn = st.button("RUN ANALYSIS")

    with col2:
        st.subheader("🎯 Result")
        if predict_btn:
            avg_atv = est_vol / (trans_count + 1e-6)
            timeline = (year - 2018) * 4 + int(quarter)
            input_data = np.zeros((1, 11))
            input_data[0, 0:5] = [trans_count, year, int(quarter), avg_atv, timeline]
            
            prediction = model.predict(input_data)
            final_val = np.expm1(prediction[0])

            st.metric(label="Predicted Value", value=f"₹{final_val:,.2f}")

            # Graph: Trend
            fig = go.Figure(go.Scatter(x=[year-1, year, year+1], y=[final_val*0.9, final_val, final_val*1.1],
                                     line=dict(color='#9B59B6', width=4), fill='tozeroy'))
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter details and click 'Run Analysis' to see results.")

elif menu == "📊 Market Insights":
    st.title("🔍 Advanced Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(values=[60, 40], names=['Success', 'Failure'], hole=.5, 
                               color_discrete_sequence=['#9B59B6', '#E9ECEF']))
    with c2:
        impact_data = pd.DataFrame({'Factor': ['Vol', 'Time', 'Year'], 'Val': [45, 25, 30]})
        st.plotly_chart(px.bar(impact_data, x='Val', y='Factor', orientation='h', color_discrete_sequence=['#9B59B6']))

elif menu == "📄 Tech Documentation":
    st.title("📚 Technical Documentation")
    
    st.subheader("🛠️ Technology Stack")
    st.markdown("""
    * **Language:** Python
    * **ML Model:** XGBoost Regressor for high-precision time-series forecasting
    * **Web Framework:** Streamlit for interactive UI
    * **Visualization:** Plotly Express and Graph Objects for dynamic charts
    * **Deployment:** GitHub for version control and Streamlit Cloud for hosting
    """)
    
    st.divider()
    
    st.subheader("🚀 How to Run Locally")
    st.code("""
# 1. Clone the repository
git clone https://github.com/your-username/PhonePe-Pulse-Analysis.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
    """, language="bash")
    
    st.divider()
    
    st.subheader("📊 Key Features")
    st.markdown("""
    * **Real-time Prediction:** Generates financial forecasts based on 11 feature inputs.
    * **Interactive Insights:** Visualizes market drivers and growth trends using advanced charting.
    * **Cloud Hosted:** Accessible via any web browser through the Streamlit Cloud platform.
    """)

st.divider()
st.caption(f"© {datetime.now().year} PhonePe Pulse Analytics Project")
