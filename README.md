# Delhi AQI Prediction App

A Streamlit web application to predict Delhi’s Air Quality Index (AQI) category and PM2.5 levels using machine learning models. Enter pollutant concentrations to get instant AQI predictions powered by Random Forest and XGBoost.

## Features

- Predicts PM2.5 concentration from user-input pollutant levels.
- Classifies AQI category using Random Forest and XGBoost models.
- User-friendly Streamlit interface.
- Trained on real Delhi air quality data.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Delhi-AQI-Prediction.git
   cd Delhi-AQI-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files and dataset are in place:**
   - Place pre-trained models (`rf_model.pkl`, `xgb_model.pkl`, `linear_model.pkl`, `label_encoder.pkl`) in the `models/` directory.
   - Place the dataset (`delhi_aqi.csv`) in the `data/` directory.

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser and go to:**  
   [http://localhost:8501](http://localhost:8501)

## Project Structure

```
Delhi-AQI-Prediction/
│
├── app.py
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── linear_model.pkl
│   └── label_encoder.pkl
├── data/
│   └── delhi_aqi.csv
└── requirements.txt
```

## Input Parameters

- **CO** (Carbon Monoxide)
- **NO** (Nitric Oxide)
- **NO2** (Nitrogen Dioxide)
- **O3** (Ozone)
- **SO2** (Sulfur Dioxide)
- **PM10** (Particulate Matter 10)
- **NH3** (Ammonia)

## Output

- **Predicted PM2.5 Level** (µg/m³)
- **Random Forest AQI Category**
- **XGBoost AQI Category**

## License

This project is for educational and research purposes.
