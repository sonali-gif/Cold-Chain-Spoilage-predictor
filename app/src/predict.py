import joblib
import pandas as pd
from pathlib import Path

# Get the project directory

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the trained model

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_spoilage(sensor_data):
    # Convert input dictionary into a DataFrame
    input_df = pd.DataFrame([sensor_data])

    # Predict the spoilage stage
    prediction = model.predict(input_df)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Get confidence percentage
    confidence = max(probabilities) * 100

    return {
        "predicted_stage": prediction,
        "confidence": round(confidence, 2),
    }

# Test the prediction function

if __name__ == "__main__":
    sensor_data = {
        "Vegetable_Type": "Tomato",
        "MQ135_ADC_Raw": 115,
        "MQ135_ADC_RollingAvg12": 112,
        "MQ135_ADC_ShortAvg3": 117,
        "R0_Baseline": 100,
        "Rs_R0_Ratio": 1.00,
        "MQ135_VOC_Index_Est": 0.25,
        "DHT22_Temp_C": 22.5,
        "DHT22_Humidity_%": 65,
        "Humidity_Baseline_Prev12": 65,
        "Humidity_Local_Spike": 0,
        "Gas_Slope_ADC_per_hr": 0.3,
        "Spoilage_Signal": 0.25,
    }

    result = predict_spoilage(sensor_data)

    print("\n========== SPOILAGE PREDICTION ==========")
    print("Vegetable:", sensor_data["Vegetable_Type"])
    print("Predicted Stage:", result["predicted_stage"])
    print("Confidence:", str(result["confidence"]) + "%")
