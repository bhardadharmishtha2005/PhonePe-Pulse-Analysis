import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PhonePe Pulse ML", layout="wide")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Project Hub")

    menu = st.radio(
        "GO TO:",
        ["🚀 Predictor Engine", "📈 Advanced Analytics", "📄 Documentation"]
    )

# ---------------- PREDICTOR ----------------
if menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Prediction Engine")

    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.subheader("Input Parameters")

        trans_count = st.number_input("Transaction Count", value=5000)
        year = st.select_slider("Select Year", options=list(range(2018, 2027)), value=2024)
        quarter = st.radio("Select Quarter", [1, 2, 3, 4], horizontal=True)
        volume = st.number_input("Regional Volume (₹)", value=150000)

        run = st.button("GENERATE AI FORECAST")

    with c2:
        st.subheader("Result")

        if run:
            pred = (volume / (trans_count + 1)) * 10
            st.metric("Predicted Value", f"₹{pred:,.2f}")

            fig = px.area(
                x=[year-1, year, year+1],
                y=[pred*0.8, pred, pred*1.2],
                title="Forecast Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- MAP ----------------
elif menu == "📈 Advanced Analytics":
    st.title("🔍 Geospatial & Market Insights")
    st.subheader("🗺️ India Transaction Heatmap")

    map_data = pd.DataFrame({
        'State': [
            'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar',
            'Chhattisgarh','Goa','Gujarat','Haryana',
            'Himachal Pradesh','Jharkhand','Karnataka','Kerala',
            'Madhya Pradesh','Maharashtra','Manipur',
            'Meghalaya','Mizoram','Nagaland','Odisha',
            'Punjab','Rajasthan','Sikkim','Tamil Nadu',
            'Telangana','Tripura','Uttar Pradesh',
            'Uttarakhand','West Bengal'
        ],
        'Value': np.random.randint(50000, 150000, 28)
    })

    try:
        geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
        india_geojson = requests.get(geojson_url).json()

        fig_map = px.choropleth(
            map_data,
            geojson=india_geojson,
            featureidkey="properties.NAME_1",
            locations="State",
            color="Value",
            color_continuous_scale="Blues"
        )

        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"Map failed: {e}")

# ---------------- DOC ----------------
elif menu == "📄 Documentation":
    st.title("📄 Documentation")
    st.write("Project details here")
