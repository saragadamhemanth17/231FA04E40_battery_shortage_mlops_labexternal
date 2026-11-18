# Battery Shortage Prediction Using MLOps Pipeline

## System Architecture
An end-to-end MLOps solution using Python, FastAPI, and Pytest to predict electric scooter swapping-station battery shortages.

```
[Synthetic CSV Generation] -> [Data Preprocessing & Feature Engineering] 
                                  |
                                  v
[FastAPI Deployments]      <- [Model Selection & Export] <- [MLflow Runs Tracked]
```

## Dataset Versioning Details
- **Version 1 (Normal Operations):** 250 records representing standard walk-in and booking request rates.
- **Version 2 (Festival Operations):** 250 records with heightened demand signals to model outlier operations.

## Feature Engineering & Formulas
- **Expected Battery Supply:** `charged_batteries_5pm + charging_completed_by_7pm`
- **Booking Pressure Ratio:** `advance_bookings / (expected_battery_supply + 1)`

## Evaluation Results
- **Selected Model:** Logistic Regression
- **Key Priority Metric (Recall):** 98.33% recall to minimize missed battery shortages.

## Running the Project Locally
1. Install packages:
   ```bash
   pip install fastapi uvicorn pandas joblib scikit-learn pytest
   ```
2. Run automated test cases:
   ```bash
   pytest test_pipeline.py -v
   ```
3. Launch the API locally:
   ```bash
   uvicorn main:app --reload --port 8000
   ```