"""
Prediction Module
-----------------
This module loads the trained model and predicts house prices.
"""

import joblib
import numpy as np

# Load Saved Files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")


def predict_house_price(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity
):
    """
    Predict House Price
    """

    # Encode categorical value
    ocean_value = encoder.transform([ocean_proximity])[0]

    # Create input array
    input_data = np.array([
        [
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_value
        ]
    ])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    return prediction[0]