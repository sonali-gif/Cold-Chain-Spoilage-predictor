from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.predict import predict_spoilage
from backend.database import init_db, save_prediction, get_predictions


app = FastAPI(
    title="Vegetable Spoilage Predictor API",
    description="API for predicting vegetable spoilage using Machine Learning",
    version="1.0.0"
)


# Initialize database
@app.on_event("startup")
def startup():
    init_db()


# Sensor input structure
class SensorData(BaseModel):
    Vegetable_Type: str
    MQ135_ADC_Raw: float
    MQ135_ADC_RollingAvg12: float
    MQ135_ADC_ShortAvg3: float
    R0_Baseline: float
    Rs_R0_Ratio: float
    MQ135_VOC_Index_Est: float
    DHT22_Temp_C: float
    DHT22_Humidity_Percent: float
    Humidity_Baseline_Prev12: float
    Humidity_Local_Spike: float
    Gas_Slope_ADC_per_hr: float
    Spoilage_Signal: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Vegetable Spoilage Predictor API is running!"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: SensorData):

    try:
        sensor_data = data.model_dump()

        sensor_data["DHT22_Humidity_%"] = sensor_data.pop(
            "DHT22_Humidity_Percent"
        )

        result = predict_spoilage(sensor_data)

        save_prediction(data.model_dump(), result)

        return {
            "vegetable": data.Vegetable_Type,
            "predicted_stage": result["predicted_stage"],
            "confidence": result["confidence"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Prediction history endpoint
@app.get("/history")
def history():

    try:
        records = get_predictions()

        return {
            "total_records": len(records),
            "history": records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))