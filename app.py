import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# Page config
st.set_page_config(page_title="PhonePe Pulse ML", layout="wide")

# ✅ SIDEBAR FIRST
with st.sidebar:
    st.title("Project Hub")

    menu = st.radio(
        "GO TO:",
        ["🚀 Predictor Engine", "📈 Advanced Analytics", "📄 Documentation"]
    )

# ✅ NOW USE menu
if menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Prediction Engine")
    st.write("Your predictor UI here")

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

elif menu == "📄 Documentation":
    st.title("📄 Documentation")
    st.write("Project details here")
