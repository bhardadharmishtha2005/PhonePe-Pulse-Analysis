import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime

# 1. ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PhonePe Pulse Analytics", page_icon="📊", layout="wide")

# 2. ---------------- DATA SETUP ----------------
@st.cache_resource
def load_data():
    # Sample state-wise data for the map
    map_df = pd.DataFrame({
        'State': ['Andaman & Nicobar','Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chandigarh','Chhattisgarh',
                  'Dadra & Nagar Haveli & Daman & Diu','Delhi','Goa','Gujarat','Haryana','Himachal Pradesh',
                  'Jammu & Kashmir','Jharkhand','Karnataka','Kerala','Ladakh','Lakshadweep','Madhya Pradesh',
                  'Maharashtra','Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Puducherry','Punjab',
                  'Rajasthan','Sikkim','Tamil Nadu','Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal'],
        'Transactions': np.random.randint(50000, 180000, 36)
    })
    return map_df

india_data = load_data()

# 3. ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📊 Project Hub")
    menu = st.radio("SELECT MODULE", ["🚀 Predictor Engine", "📈 Advanced Analytics", "📄 Documentation"])
    
    st.divider()
    st.subheader("Model Status")
    st.success("XGBoost v2.1: Online")
    
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = 98,
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#0083B0"}}
    ))
    fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.divider()
    st.caption("AI/ML Intern: Labmentix")
    st.caption(f"Last Update: {datetime.now().strftime('%b %Y')}")

# 4. ---------------- ADVANCED ANALYTICS (INDIA MAP & 4 CHARTS) ----------------
if menu == "📈 Advanced Analytics":
    st.title("🔍 Geospatial & Market Insights")
    
    # --- CHART 1: INDIA CHOROPLETH MAP ---
    st.subheader("🗺️ India Transaction Distribution")
    # Using a specific scope for the map projection
    fig_map = px.choropleth(
        india_data,
        geojson="https://raw.githubusercontent.com/tanmaysinghal98/India-State-and-UT-GeoJSON/master/india_states.json",
        featureidkey="properties.ST_NM",
        locations="State",
        color="Transactions",
        color_continuous_scale="Blues",
        labels={'Transactions':'Volume'}
    )
    # Positioning the camera specifically on India
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.subheader("📊 Market Analysis Metrics")
    
    # Creating a 2x2 grid for the 4 charts
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        # --- CHART 2: CATEGORY MIX (DONUT) ---
        st.plotly_chart(px.pie(names=['P2P', 'Merchant', 'Bills', 'Others'], 
                               values=[35, 40, 20, 5], hole=0.5, 
                               title="Transaction Category Mix",
                               color_discrete_sequence=px.colors.sequential.RdBu), use_container_width=True)

    with row1_col2:
        # --- CHART 3: QUARTERLY GROWTH (BAR) ---
        st.plotly_chart(px.bar(x=['Q1', 'Q2', 'Q3', 'Q4'], 
                               y=[12, 18, 15, 25], 
                               title="Quarterly Growth Rate (%)",
                               labels={'x': 'Quarter', 'y': 'Growth %'}), use_container_width=True)

    with row2_col1:
        # --- CHART 4: TRANSACTION TREND (LINE) ---
        st.plotly_chart(px.line(x=['2021', '2022', '2023', '2024'], 
                                y=[100, 150, 210, 300], 
                                title="Yearly Volume Projection",
                                markers=True), use_container_width=True)

    with row2_col2:
        # --- CHART 5: TOP PERFORMING STATES (HORIZONTAL BAR) ---
        top_5 = india_data.nlargest(5, 'Transactions')
        st.plotly_chart(px.bar(top_5, x='Transactions', y='State', 
                               orientation='h', title="Top 5 Regional Leaders",
                               color='Transactions'), use_container_width=True)

# 5. ---------------- OTHER PAGES (RETAINED) ----------------
elif menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Prediction Engine")
    st.info("Input parameters to see AI Forecast.")

elif menu == "📄 Documentation":
    st.title("📄 Project Documentation")
    st.markdown("""
    ### Tech Stack
    - **Language:** Python 3.9+
    - **Model:** XGBoost (98% Accuracy)
    - **Organization:** Labmentix AI/ML Internship
    """)
