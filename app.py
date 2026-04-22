if menu == "🚀 Predictor Engine":
    st.title("⚡ Transaction Prediction Engine")

    c1, c2 = st.columns([1, 1.5], gap="large")

    with c1:
        st.subheader("Input Parameters")

        trans_count = st.number_input("Transaction Count", value=5000)
        year = st.select_slider("Select Year", options=list(range(2018, 2027)), value=2024)
        quarter = st.radio("Select Quarter", [1, 2, 3, 4], horizontal=True)
        volume = st.number_input("Regional Volume (₹)", value=150000)

        run = st.button("GENERATE AI FORECAST")

    with c2:
        st.subheader("Intelligence Result")

        if run:
            pred = (volume / (trans_count + 1)) * 10  # dummy logic

            st.metric("Predicted Transaction Value", f"₹{pred:,.2f}")

            fig = px.area(
                x=[year-1, year, year+1],
                y=[pred*0.8, pred, pred*1.2],
                title="Forecast Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
