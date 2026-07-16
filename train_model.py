# ==========================================
# House Price Prediction - Model Training
# ==========================================

import pandas as pd
import numpy as np
import joblib
import json
# Custom Preprocessing Functions
from utils.preprocess import (
    clean_data,
    encode_data,
    split_features_target
)

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("housing.csv")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(df.head())

# ==========================================
# Check Missing Values
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# Data Cleaning
# ==========================================

df = clean_data(df)

print("\nData Cleaning Completed Successfully")

# ==========================================
# Encode Categorical Data
# ==========================================

df, encoder = encode_data(df)

print("Categorical Encoding Completed Successfully")

# ==========================================
# Features and Target
# ==========================================

X, y = split_features_target(df)

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain-Test Split Completed")

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# ==========================================
# Train Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Model Evaluation
# ==========================================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

# ==========================================
# Adjusted R²
# ==========================================

n = X.shape[0]

p = X.shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

# ==========================================
# Print Metrics
# ==========================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"R² Score      : {r2:.4f}")
print(f"Adjusted R²   : {adjusted_r2:.4f}")
print(f"MAE           : {mae:.2f}")
print(f"MSE           : {mse:.2f}")
print(f"RMSE          : {rmse:.2f}")

# ==========================================
# Model Parameters
# ==========================================

print("\n" + "=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)

print(f"Intercept : {model.intercept_}")

print("\nCoefficients")

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:25} : {coef}")

# ==========================================
# Save Model
# ==========================================
# ==========================================
# Save Evaluation Metrics
# ==========================================

metrics = {
    "r2_score": float(r2),
    "adjusted_r2": float(adjusted_r2),
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("✅ Metrics Saved Successfully")


joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\n" + "=" * 60)
print("Model Saved Successfully")
print("Files Created:")
print("✔ model.pkl")
print("✔ scaler.pkl")
print("✔ encoder.pkl")
print("=" * 60)