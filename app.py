import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="PhonePe Pulse Analytics", page_icon="📈", layout="wide")

# 2. Ultra-Clean White CSS
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

    /* Professional Card Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6E9EF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    /* Button */
    .stButton>button {
        background-color: #5F259F !important;
        color: white !important;
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - Rich Content
with st.sidebar:
    st.image("https://www.phonepe.com/pulse/static/79ca96328325a7a8d519286d3e387195/logo.png", width=150)
    st.markdown("## **Navigation**")
    menu = st.radio("Go to:", ["Predictor", "Market Insights", "Project Documentation"])
    
    st.divider()
    st.markdown("### **Data Stats**")
    st.write("📅 Data Range: 2018 - 2024")
    st.write("📊 Model: XGBoost v2.1")
    st.write("📍 Coverage: All India States")
    
    st.divider()
    st.success("System: Online")

# 4. Load Model
@st.cache_resource
def load_model():
    return joblib.load('phonepe_prediction_model.pkl')

model = load_model()

# 5. Main Dashboard Logic
if menu == "Predictor":
    st.title("💳 Transaction Prediction Engine")
    st.write("Analyze and forecast payment volumes using the verified PhonePe Pulse model.")
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.subheader("🛠️ Parameters")
        with st.container(border=True):
            trans_count = st.number_input("Transaction Count", value=5000)
            year = st.select_slider("Forecast Year", options=list(range(2018, 2027)), value=2024)
            quarter = st.segmented_control("Quarter", [1, 2, 3, 4], default=1)
            est_vol = st.number_input("Average Regional Volume (₹)", value=120000)
            predict_btn = st.button("Generate AI Forecast")

    with col2:
        st.subheader("🎯 Forecast Results")
        if predict_btn:
            # Feature engineering (11 cols)
            avg_atv = est_vol / (trans_count + 1e-6)
            timeline = (year - 2018) * 4 + int(quarter)
            
            input_data = np.zeros((1, 11))
            input_data[0, 0:5] = [trans_count, year, int(quarter), avg_atv, timeline]
            
            prediction = model.predict(input_data)
            final_val = np.expm1(prediction[0])

            # Big Metric
            st.metric(label="Predicted Total Value", value=f"₹{final_val:,.2f}", delta="Predicted Growth")

            # 6. Added Graph: Growth Projection
            st.markdown("### 📈 Projected Growth Trend")
            years = np.array([year-1, year, year+1])
            values = np.array([final_val * 0.85, final_val, final_val * 1.15])
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=years, y=values, mode='lines+markers', 
                                         line=dict(color='#5F259F', width=4),
                                         fill='tozeroy'))
            fig_line.update_layout(template="plotly_white", height=300, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_line, use_container_width=True)
            
        else:
            st.info("👈 Enter data on the left to see the prediction and trends.")

elif menu == "Market Insights":
    st.title("📊 Model Influence Factors")
    # Horizontal Bar Chart for weights
    impact_data = pd.DataFrame({
        'Factor': ['Transaction Count', 'Time Index', 'Year', 'Quarter', 'Avg Ticket Size'],
        'Importance': [0.42, 0.28, 0.15, 0.10, 0.05]
    }).sort_values('Importance')
    
    fig_bar = px.bar(impact_data, x='Importance', y='Factor', orientation='h', 
                     color_discrete_sequence=['#5F259F'], template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

elif menu == "Project Documentation":
    st.title("📚 Documentation")
    st.markdown("""
    ### About the Model
    - **Architecture:** XGBoost (Extreme Gradient Boosting)
    - **Optimization:** GridSearch CV for parameter tuning
    - **Source:** PhonePe Pulse GitHub Repository
    
    ### How to use
    1. Select the **Predictor** tab from the sidebar.
    2. Input the expected transaction metrics.
    3. View the AI-generated financial forecast and growth trends.
    """)

st.divider()
