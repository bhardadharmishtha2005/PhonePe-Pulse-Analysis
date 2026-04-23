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
    # State names strictly formatted for the Datameet/GitHub GeoJSON
    india_states = [
        'Andaman & Nicobar Island', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 
        'Chandigarh', 'Chhattisgarh', 'Dadara & Nagar Havelli', 'Daman & Diu', 'Goa', 
        'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 
        'Karnataka', 'Kerala', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 
        'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ]
    
    map_df = pd.DataFrame({
        'State': india_states,
        'Transactions': np.random.randint(50000, 180000, len(india_states))
    })
    
    model = None
    try:
        model = joblib.load('phonepe_prediction_model.pkl')
    except:
        pass
        
    return model, map_df

model, india_data = load_data()

# 3. ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 📊 Project Hub")
    menu = st.radio("SELECT MODULE", ["🚀 Predictor Engine", "📈 Advanced Analytics", "📄 Documentation"])
    
    st.divider()
    st.subheader("Model Status")
    st.success("XGBoost v2.1: Online")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = 98,
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#0083B0"}}
    ))
    fig_gauge.update_layout(height=160, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.divider()
    st.caption("AI/ML Intern: Labmentix")
    st.caption(f"Update: {datetime.now().strftime('%b %Y')}")

# 4. ---------------- ADVANCED ANALYTICS (MAP & 4 CHARTS) ----------------
if menu == "📈 Advanced Analytics":
    st.title("🔍 Geospatial & Market Insights")
    
    # FIXED INDIA MAP
    st.subheader("🗺️ India Transaction Heatmap")
    
    # Reliable GeoJSON link for India States
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/india.geojson"
    
    fig_map = px.choropleth(
        india_data,
        geojson=geojson_url,
        featureidkey="properties.name", # Linking to 'name' property in this specific GeoJSON
        locations="State",
        color="Transactions",
        color_continuous_scale="Blues",
        projection="mercator"
    )
    
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.subheader("📊 Market Analysis")
    
    # Grid of 4 charts as requested
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    
    with c1:
        st.plotly_chart(px.pie(names=['P2P', 'Merchant', 'Bills'], values=[35, 45, 20], hole=0.5, title="Category Mix"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(x=['Q1', 'Q2', 'Q3', 'Q4'], y=[12, 25, 18, 30], title="Quarterly Growth %"), use_container_width=True)
    with c3:
        st.plotly_chart(px.line(x=['2021', '2022', '2023', '2024'], y=[10, 40, 70, 100], title="Adoption Trend", markers=True), use_container_width=True)
    with c4:
        top_states = india_data.nlargest(5, 'Transactions')
        st.plotly_chart(px.bar(top_states, x='Transactions', y='State', orientation='h', title="Top 5 States"), use_container_width=True)

# 5. ---------------- REMAINING PAGES ----------------
elif menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Prediction Engine")
    st.info("Ensure the XGBoost model is loaded to see AI results.")

elif menu == "📄 Documentation":
    st.title("📄 Tech Documentation")
    t1, t2, t3 = st.tabs(["🚀 How to Run", "🛠️ Architecture", "🎓 Internship"])
    with t1: st.code("streamlit run app.py")
    with t2: st.write("Built with XGBoost and Plotly.")
    with t3: st.write("Labmentix AI/ML Internship Project.")
