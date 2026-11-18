from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(title="Battery Shortage Prediction API", version="1.0")

# Load the registered best model
MODEL_PATH = "best_battery_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Define the Input Schema with Validation Rules
class StationCondition(BaseModel):
    charged_batteries_5pm: int = Field(..., ge=0, description="Number of charged batteries available at 5:00 PM")
    charging_completed_by_7pm: int = Field(..., ge=0, description="Number of batteries completing charge by 7:00 PM")
    advance_bookings: int = Field(..., ge=0, description="Number of booked battery swaps")
    walk_in_demand: int = Field(..., ge=0, description="Historical walk-in customer demand")
    rainfall_forecast_mm: float = Field(..., ge=0.0, description="Rainfall forecast in mm")
    weekday_weekend: int = Field(..., ge=0, le=1, description="0 for weekday, 1 for weekend")
    nearby_event: int = Field(..., ge=0, le=1, description="0 for no nearby event, 1 for nearby event")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the Battery Shortage Prediction API"}

@app.post("/predict")
def predict_shortage(data: StationCondition):
    # Extract fields from payload
    charged_5pm = data.charged_batteries_5pm
    charging_7pm = data.charging_completed_by_7pm
    bookings = data.advance_bookings
    
    # Re-apply matching feature engineering logic
    expected_supply = charged_5pm + charging_7pm
    booking_ratio = bookings / (expected_supply + 1)
    
    # Prepare features in exact order used during training
    input_features = pd.DataFrame([{
        'charged_batteries_5pm': charged_5pm,
        'charging_completed_by_7pm': charging_7pm,
        'advance_bookings': bookings,
        'walk_in_demand': data.walk_in_demand,
        'rainfall_forecast_mm': data.rainfall_forecast_mm,
        'weekday_weekend': data.weekday_weekend,
        'nearby_event': data.nearby_event,
        'expected_battery_supply': expected_supply,
        'booking_pressure_ratio': round(booking_ratio, 4)
    }])
    
    # Predict
    prediction = int(model.predict(input_features)[0])
    result = "Shortage Expected" if prediction == 1 else "Sufficient Batteries"
    
    return {
        "prediction_code": prediction,
        "status": result,
        "computed_features": {
            "expected_battery_supply": expected_supply,
            "booking_pressure_ratio": round(booking_ratio, 4)
        }
    }