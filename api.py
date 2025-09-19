from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Will it rain tomorrow API", description="API za predikciju padavina u Australiji", version="1.0")

# Učitavanje modela
MODEL_PATH = 'best_xgboost_pipeline.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Definiši ulazne podatke (prilagodi prema stvarnim kolonama)
class PredictionInput(BaseModel):
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustSpeed: float
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Temp9am: float
    Temp3pm: float
    Cloud9am: float
    Cloud3pm: float
    RainToday: str = Field(..., regex='^(Yes|No)$')
    WindGustDir: str
    WindDir9am: str
    WindDir3pm: str
    Location: str
    Month: int = Field(..., ge=1, le=12)
    Day: int = Field(..., ge=1, le=31)

@app.post('/predict')
def predict(input: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail='Model nije dostupan!')
    try:
        input_df = pd.DataFrame([input.dict()])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        return {
            'prediction': 'DA' if pred == 'Yes' else 'NE',
            'probability': round(float(proba), 3)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Greška u predikciji: {e}')

# Automatska Swagger dokumentacija dostupna na /docs
