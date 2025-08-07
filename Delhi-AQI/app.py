import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
rf_model_path = os.path.join(current_dir, 'models', 'rf_model.pkl')
xgb_model_path = os.path.join(current_dir, 'models', 'xgb_model.pkl')
label_encoder_path = os.path.join(current_dir, 'models', 'label_encoder.pkl')
linear_model_path = os.path.join(current_dir, 'models', 'linear_model.pkl')

# Load models
rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)
linear_model = joblib.load(linear_model_path)
label_encoder = joblib.load(label_encoder_path)

# Load dataset
data = pd.read_csv(os.path.join(current_dir, 'data', 'delhi_aqi.csv'))

# Streamlit UI
st.title("Air Quality Index (AQI) Prediction")

# Input section
st.header("Enter Pollutant Levels (in µg/m³):")
co = st.number_input("CO", min_value=0.0, value=1.0)
no = st.number_input("NO", min_value=0.0, value=1.0)
no2 = st.number_input("NO2", min_value=0.0, value=1.0)
o3 = st.number_input("O3", min_value=0.0, value=1.0)
so2 = st.number_input("SO2", min_value=0.0, value=1.0)
pm10 = st.number_input("PM10", min_value=0.0, value=1.0)
nh3 = st.number_input("NH3", min_value=0.0, value=1.0)

# Prediction
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[co, no, no2, o3, so2, pm10, nh3]])

    # Predict PM2.5
    predicted_pm25 = linear_model.predict(input_data)[0]

    # Predict AQI category using Random Forest
    rf_prediction = rf_model.predict(input_data)
    rf_aqi_category = label_encoder.inverse_transform(rf_prediction)[0]

    # Predict AQI category using XGBoost
    xgb_prediction = xgb_model.predict(input_data)
    xgb_aqi_category = label_encoder.inverse_transform(xgb_prediction)[0]

    # Display results
    st.subheader("Prediction Results:")
    st.write(f"**Predicted PM2.5 Level:** {predicted_pm25:.2f} µg/m³")
    st.write(f"**Random Forest AQI Category:** {rf_aqi_category}")
    st.write(f"**XGBoost AQI Category:** {xgb_aqi_category}")
