import requests

elif menu == "📈 Advanced Analytics":
    st.title("🔍 Geospatial & Market Insights")
    st.subheader("🗺️ India Transaction Heatmap")

    # Data
    map_data = pd.DataFrame({
        'State': [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
            'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
            'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
            'Madhya Pradesh', 'Maharashtra', 'Manipur',
            'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
            'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
            'Telangana', 'Tripura', 'Uttar Pradesh',
            'Uttarakhand', 'West Bengal'
        ],
        'Value': np.random.randint(50000, 150000, 28)
    })

    map_data['State'] = map_data['State'].str.strip()

    try:
        # Load GeoJSON from working source
        geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
        india_geojson = requests.get(geojson_url).json()

        fig_map = px.choropleth(
            map_data,
            geojson=india_geojson,
            featureidkey="properties.NAME_1",  # 🔥 IMPORTANT
            locations="State",
            color="Value",
            color_continuous_scale="Blues"
        )

        fig_map.update_geos(fitbounds="locations", visible=False)

        fig_map.update_traces(
            marker_line_width=0.5,
            marker_line_color="white"
        )

        fig_map.update_layout(
            height=600,
            margin={"r":0,"t":0,"l":0,"b":0}
        )

        st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"Map failed: {e}")
        st.bar_chart(map_data.set_index('State'))
