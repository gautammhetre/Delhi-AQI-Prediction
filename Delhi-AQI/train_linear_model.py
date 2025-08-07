import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data/delhi_aqi.csv")  # Ensure this file exists in the project directory

# Prepare features and target
X = data[['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']]
y = data['pm2_5']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Save the model
joblib.dump(linear_model, "models/linear_model.pkl")

print("Linear Regression model trained and saved as models/linear_model.pkl")
