import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load Data
data = pd.read_csv('data/delhi_aqi.csv')

# Preprocessing
data = data.dropna()  # Drop rows with missing values
X = data[['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']]
y = data['pm2_5'].apply(lambda x: 'Good' if x <= 50 else
                                  'Moderate' if x <= 100 else
                                  'Unhealthy for Sensitive Groups' if x <= 150 else
                                  'Unhealthy' if x <= 200 else
                                  'Very Unhealthy' if x <= 300 else
                                  'Hazardous')

# Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost Model
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Save Models and Label Encoder
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
