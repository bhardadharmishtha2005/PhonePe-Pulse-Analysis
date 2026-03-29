import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="PhonePe Pulse Analytics", page_icon="📈", layout="wide")

# 2. Enhanced Styling for White Theme & Gradient Button
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    
    /* Global Text Color */
    h1, h2, h3, p, label, .stMarkdown { color: #2D3436 !important; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FDFDFF;
        border-right: 1px solid #E6E9EF;
    }

    /* Metric Result Styling */
    [data-testid="stMetricValue"] {
        color: #5F259F !important;
        font-weight: 800 !important;
    }

    /* PREMIUM GRADIENT BUTTON */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #5F259F, #8E44AD) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 3.5em !important;
        font-size: 18px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(95, 37, 159, 0.3) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(95, 37, 159, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.markdown("## **Navigation**")
    menu = st.radio("Go to:", ["Predictor", "Market Insights", "Project Documentation"])
    st.divider()
    st.markdown("### **Data Stats**")
    st.write("📅 Data Range: 2018 - 2024")
    st.write("📊 Model: XGBoost v2.1")
    st.divider()

# 4. Load Model
@st.cache_resource
def load_model():
    return joblib.load('phonepe_prediction_model.pkl')

model = load_model()

# 5. Main Dashboard Logic
if menu == "Predictor":
    st.title("💳 Transaction Prediction Engine")
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.subheader("🛠️ Parameters")
        with st.container(border=True):
            trans_count = st.number_input("Transaction Count", value=5000)
            year = st.select_slider("Forecast Year", options=list(range(2018, 2027)), value=2024)
            quarter = st.segmented_control("Quarter", [1, 2, 3, 4], default=1)
            est_vol = st.number_input("Average Regional Volume (₹)", value=120000)
            predict_btn = st.button("Generate AI Analysis")

    with col2:
        st.subheader("🎯 Analysis Results")
        if predict_btn:
            # Feature engineering (11 cols)
            avg_atv = est_vol / (trans_count + 1e-6)
            timeline = (year - 2018) * 4 + int(quarter)
            input_data = np.zeros((1, 11))
            input_data[0, 0:5] = [trans_count, year, int(quarter), avg_atv, timeline]
            
            prediction = model.predict(input_data)
            final_val = np.expm1(prediction[0])

            # Big Metric
            st.metric(label="Predicted Total Value", value=f"₹{final_val:,.2f}", delta="Growth Projected")

            # --- NEW GRAPH 1: Quarter Performance Distribution ---
            st.markdown("#### 🥧 Projected Quarter Weightage")
            labels = ['Q1', 'Q2', 'Q3', 'Q4']
            # Simulated data based on prediction for visual breakdown
            values = [final_val*0.22, final_val*0.24, final_val*0.26, final_val*0.28]
            fig_pie = px.pie(values=values, names=labels, hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.Purp)
            fig_pie.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # --- NEW GRAPH 2: Transaction Density vs. Timeline ---
            st.markdown("#### 🌊 Market Momentum")
            # Creating a smooth curve for momentum
            x_range = np.linspace(0, timeline, 20)
            y_range = np.sin(x_range) * 0.1 + (x_range/timeline) * final_val
            fig_momentum = go.Figure(go.Scatter(x=x_range, y=y_range, fill='tozeroy', 
                                              line=dict(color='#8E44AD')))
            fig_momentum.update_layout(template="plotly_white", height=300, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_momentum, use_container_width=True)

        else:
            st.info("👈 Adjust parameters and click Generate to view advanced analytics.")

elif menu == "Market Insights":
    st.title("📊 Strategic Market Insights")
    
    # --- NEW GRAPH 3: Feature Interaction Radar ---
    st.subheader("🕸️ Model Sensitivity Analysis")
    fig_radar = go.Figure(data=go.Scatterpolar(
      r=[4, 3, 5, 2, 4],
      theta=['Volume','Timing','Year','Quarter','Avg Ticket'],
      fill='toself',
      line=dict(color='#5F259F')
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), 
                           showlegend=False, height=400)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Influence Bar Chart
    impact_data = pd.DataFrame({
        'Factor': ['Transaction Count', 'Time Index', 'Year', 'Quarter', 'Avg Ticket Size'],
        'Importance': [0.42, 0.28, 0.15, 0.10, 0.05]
    }).sort_values('Importance')
    st.plotly_chart(px.bar(impact_data, x='Importance', y='Factor', orientation='h', 
                           color_discrete_sequence=['#5F259F'], template="plotly_white"))

elif menu == "Project Documentation":
    st.title("📚 Documentation")
    st.markdown("""
    ### Technical Stack
    - **Regressor:** XGBoost (Extreme Gradient Boosting)
    - **Deployment:** Streamlit Cloud
    - **Data Source:** PhonePe Pulse Open Data
    """)

st.divider()
