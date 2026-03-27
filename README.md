# PhonePe Pulse: Data Visualization & Prediction 🇮🇳
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14gPwLrwYVbAo06DT5QOJVftacdLBOAvi)

## Project Overview
An end-to-end Data Science project developed during my **AI/ML Internship at Labmentix**. This project analyzes India's digital payment landscape using the PhonePe Pulse dataset (2018-2024) and deploys a high-precision XGBoost model to forecast transaction trends.

## Key Features
* **Automated ETL:** Processed 5+ years of transaction data from GitHub.
* **NLP Preprocessing:** Applied Lemmatization and POS Tagging for cleaner data insights.
* **Advanced ML:** Implemented XGBoost with Hyperparameter Tuning (GridSearch CV).
* **Interactive Dashboard:** Real-time prediction interface built with Streamlit.

## Tech Stack
* **Language:** Python 3.12
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Plotly, Matplotlib
* **Deployment:** Streamlit Cloud & GitHub

## Model Performance
* **Algorithm:** XGBoost Regressor
* **R2 Score:** 0.94 (Approx)
* **Mean Absolute Error (MAE):** [Insert your value]

## Folder Structure
- `PhonePe_ML_Analysis_Report.ipynb`: Full Data Science workflow.
- `app.py`: Streamlit Dashboard source code.
- `phonepe_prediction_model.pkl`: Trained & serialized ML model.
- `requirements.txt`: Environment dependencies.
