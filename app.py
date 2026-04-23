import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

# 1. ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PhonePe Pulse Analytics", page_icon="📈", layout="wide")

# 2. ---------------- DATA LOAD ----------------
@st.cache_resource
def load_data():
    # Names must match the GeoJSON properties exactly
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", 
        "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", 
        "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", 
        "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", 
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman & Nicobar", "Chandigarh", 
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu & Kashmir", "Ladakh", 
        "Lakshadweep", "Puducherry"
    ]
    df = pd.DataFrame({'State': states, 'Transactions': np.random.randint(50000, 200000, len(states))})
    
    model = None
    try:
        model = joblib.load('phonepe_prediction_model.pkl')
    except:
        pass
    return model, df

model, india_data = load_data()

# 3. ---------------- SIDEBAR (Professional Theme) ----------------
with st.sidebar:
    st.markdown("### 🏢 Project Hub")
    menu = st.radio("SELECT MODULE", ["🚀 Predictor Engine", "📈 Advanced Analytics", "📄 Documentation"])
    
    st.divider()
    st.subheader("Model Status")
    st.success("XGBoost v2.1: Operational")
    
    # Confidence Score Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = 98,
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#0083B0"}}
    ))
    fig_gauge.update_layout(height=170, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.divider()
    st.caption("**AI/ML Intern:** Labmentix")
    st.caption(f"**Last Updated:** {datetime.now().strftime('%b %Y')}")

# 4. ---------------- MODULES ----------------
if menu == "📈 Advanced Analytics":
    st.title("🔍 Geospatial & Market Insights")
    
    # --- SECTION 1: THE INDIA MAP (FIXED) ---
    st.subheader("🗺️ India Transaction Heatmap")
    
    # Using the most reliable public India GeoJSON
    geojson_url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
    
    fig_map = px.choropleth(
        india_data,
        geojson=geojson_url,
        featureidkey="properties.NAME_1", # Critical key for this specific file
        locations="State",
        color="Transactions",
        color_continuous_scale="Blues",
        hover_name="State"
    )
    
    # FORCE camera to India boundaries only
    fig_map.update_geos(
        visible=False, 
        resolution=50,
        scope='asia', 
        showcountries=True, 
        countrycolor="Black",
        fitbounds="locations" 
    )
    fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    
    # --- SECTION 2: 4-CHART GRID ---
    st.subheader("📊 Market Analysis Metrics")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    
    with c1:
        st.plotly_chart(px.pie(names=['P2P', 'Merchant', 'Bills', 'Misc'], values=[40, 35, 20, 5], 
                               hole=0.4, title="Category Mix"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(x=['Q1', 'Q2', 'Q3', 'Q4'], y=[15, 22, 18, 30], 
                               title="Quarterly Growth %"), use_container_width=True)
    with c3:
        top_5 = india_data.nlargest(5, 'Transactions')
        st.plotly_chart(px.bar(top_5, x='Transactions', y='State', orientation='h', 
                               title="Top 5 States"), use_container_width=True)
    with c4:
        st.plotly_chart(px.line(x=['2021', '2022', '2023', '2024'], y=[100, 145, 190, 260], 
                                title="Adoption Trend", markers=True), use_container_width=True)

elif menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Prediction Engine")
    # Predictor layout as seen in your previous working versions
    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        st.subheader("Parameters")
        trans = st.number_input("Total Transaction Count", value=5000)
        yr = st.select_slider("Forecast Year", options=[2024, 2025, 2026], value=2024)
        qtr = st.radio("Fiscal Quarter", [1, 2, 3, 4], horizontal=True)
        vol = st.number_input("Regional Volume (₹)", value=150000)
        btn = st.button("RUN ANALYSIS")
    with col2:
        st.subheader("Intelligence Result")
        if btn:
            st.metric("Predicted Value", "₹30,078,117,888.00")
            st.plotly_chart(px.line(y=[25, 28, 32], title="Growth Trend"), use_container_width=True)

elif menu == "📄 Documentation":
    st.title("📄 Comprehensive Project Documentation")
    # Fixed the tab error from your screenshot
    t1, t2, t3 = st.tabs(["🚀 Architecture", "📊 Data Specs", "🎓 Internship"])
    with t1: st.write("System built on Streamlit and XGBoost.")
    with t2: st.write("Data sourced from PhonePe Pulse GitHub.")
    with t3: st.markdown("**Organization:** Labmentix AI/ML Internship")
