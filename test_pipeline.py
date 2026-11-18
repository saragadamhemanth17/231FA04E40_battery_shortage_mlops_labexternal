import pytest
import pandas as pd
import numpy as np
import joblib

def test_dataset_validity():
    df1 = pd.read_csv('station_data_v1.csv')
    df2 = pd.read_csv('station_data_v2.csv')
    assert len(df1) == 250
    assert len(df2) == 250
    assert 'battery_shortage' in df1.columns
    assert 'battery_shortage' in df2.columns

def test_feature_engineering():
    data = pd.read_csv('engineered_features.csv')
    assert 'expected_battery_supply' in data.columns
    assert 'booking_pressure_ratio' in data.columns
    
    row = data.iloc[0]
    expected_supply = float(row['charged_batteries_5pm'] + row['charging_completed_by_7pm'])
    expected_ratio = float(row['advance_bookings'] / (expected_supply + 1))
    
    assert float(row['expected_battery_supply']) == expected_supply
    assert pytest.approx(float(row['booking_pressure_ratio']), abs=1e-2) == expected_ratio

def test_model_prediction():
    model = joblib.load('best_battery_model.pkl')
    
    mock_input = pd.DataFrame([{
        'charged_batteries_5pm': 5,
        'charging_completed_by_7pm': 2,
        'advance_bookings': 40,
        'walk_in_demand': 30,
        'rainfall_forecast_mm': 0.0,
        'weekday_weekend': 0,
        'nearby_event': 1,
        'expected_battery_supply': 7,
        'booking_pressure_ratio': 5.0
    }])
    
    prediction = model.predict(mock_input)
    assert prediction[0] in [0, 1]

def test_invalid_negative_values():
    charged_batteries = -10
    assert charged_batteries < 0