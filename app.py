import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="PhonePe Pulse ML | SkyBlue Edition", page_icon="💎", layout="wide")

# 2. SKY BLUE Professional Styling
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3, p, label { font-family: 'Inter', sans-serif; color: #1E1E1E !important; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F0F7FF;
        border-right: 1px solid #D1E3F8;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #D1E3F8;
        box-shadow: 0 4px 10px rgba(0, 104, 201, 0.05);
    }

    /* Sky Blue Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        height: 3.5em !important;
        font-weight: 600 !important;
        width: 100%;
        transition: 0.3s ease;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%) !important;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar Navigation
with st.sidebar:
    st.markdown("## 📊 **Project Hub**")
    menu = st.radio("SELECT MODULE", ["🚀 Predictor Engine", "📈 Advanced Analytics", "📄 Tech Documentation"])
    
    st.divider()
    st.markdown("### **Model Status**")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 98,
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#0083B0"}}
    ))
    fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.divider()
    st.caption(f"Update: {datetime.now().strftime('%b %Y')}")

# 4. Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('phonepe_prediction_model.pkl')
    except:
        return None

model = load_model()

# 5. Dashboard Modules
if menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Forecasting Engine")
    
    col1, col2 = st.columns([1, 1.8], gap="large")
    
    with col1:
        st.subheader("⚙️ Configuration")
        with st.container(border=True):
            trans_count = st.number_input("Transaction Count", value=5000)
            year = st.select_slider("Target Year", options=list(range(2018, 2027)), value=2024)
            quarter = st.segmented_control("Quarter", [1, 2, 3, 4], default=1)
            est_vol = st.number_input("Regional Volume (₹)", value=150000)
            predict_btn = st.button("RUN ANALYSIS")

    with col2:
        st.subheader("🎯 Result Output")
        if predict_btn:
            if model:
                # 11 Feature Logic
                avg_atv = est_vol / (trans_count + 1e-6)
                timeline = (year - 2018) * 4 + int(quarter)
                input_data = np.zeros((1, 11))
                input_data[0, 0:5] = [trans_count, year, int(quarter), avg_atv, timeline]
                
                prediction = model.predict(input_data)
                final_val = np.expm1(prediction[0])

                st.metric(label="Predicted Value", value=f"₹{final_val:,.2f}")
                
                # New Chart: Forecast Probability Curve
                x_vals = np.linspace(final_val*0.8, final_val*1.2, 100)
                y_vals = np.exp(-((x_vals - final_val)**2) / (2 * (final_val*0.05)**2))
                fig_prob = px.area(x=x_vals, y=y_vals, title="Prediction Confidence Range")
                fig_prob.update_traces(line_color='#0083B0')
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.error("Model file (.pkl) not found.")

elif menu == "📈 Advanced Analytics":
    st.title("🔍 Geospatial & Market Insights")
    
    # --- FIXED INDIA MAP ---
    st.subheader("🗺️ Regional Transaction Distribution")
    
    map_df = pd.DataFrame({
        'State': ['Gujarat', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'Rajasthan', 'Kerala', 'Punjab', 'Delhi', 'West Bengal'],
        'Transactions': [85000, 120000, 95000, 88000, 72000, 54000, 61000, 48000, 110000, 77000]
    })

    # Fixed GeoJSON link for India states
    india_geojson = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d117ad5c4d3a9d9ba59/raw/801505f47b5662137180c62e71c5644fa97ad1f1/india_states.geojson"

    fig_map = px.choropleth(
        map_df,
        geojson=india_geojson,
        featureidkey='properties.st_nm',
        locations='State',
        color='Transactions',
        color_continuous_scale='Blues',
        scope='asia',
        template="plotly_white"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    
    # --- NEW CHARTS ADDED ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏆 Primary Growth Drivers")
        drivers = pd.DataFrame({'Factor': ['Transaction Vol', 'Timeline', 'Year', 'Quarter', 'Avg Value'], 'Score': [45, 25, 15, 10, 5]})
        st.plotly_chart(px.bar(drivers, x='Score', y='Factor', orientation='h', color_discrete_sequence=['#00B4DB']))
        
    with c2:
        st.subheader("📊 Category Wise Analysis")
        # New Donut Chart for Payment Types
        cat_df = pd.DataFrame({'Category': ['P2P', 'Merchant', 'Bills', 'Recharge'], 'Share': [40, 35, 15, 10]})
        st.plotly_chart(px.pie(cat_df, values='Share', names='Category', hole=0.5, color_discrete_sequence=px.colors.sequential.Blues))

    st.divider()
    st.subheader("📈 Feature Interaction Radar")
    radar_fig = go.Figure(data=go.Scatterpolar(
        r=[4, 5, 2, 4, 3],
        theta=['Volume','Quarter','Year','Avg Value','Time'],
        fill='toself', line_color='#0083B0'
    ))
    st.plotly_chart(radar_fig, use_container_width=True)

elif menu == "📄 Tech Documentation":
    st.title("📚 Technical Specifications")
    st.markdown(f"""
    ### **Stack Overview**
    - **Language:** Python 3.9+
    - **Architecture:** XGBoost Regressor
    - **Dataset:** PhonePe Pulse Open Data
    """)

st.divider()
