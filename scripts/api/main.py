from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Load model and feature names
MODEL_PATH = "models/RandomForest_model.pkl"
FEATURE_NAMES_PATH = "models/feature_names.txt"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
    raise RuntimeError("Model or feature names file not found. Please train the model first.")

model = joblib.load(MODEL_PATH)
with open(FEATURE_NAMES_PATH, "r") as f:
    feature_names = [line.strip() for line in f]

# Define FastAPI app
app = FastAPI(title="Credit Risk Prediction API")

# Define input schema
class CreditFeatures(BaseModel):
    values: list[float]  # ordered list of feature values

@app.get("/")
def root():
    return {"message": "Welcome to the Credit Risk Prediction API!"}

@app.post("/predict")
def predict_risk(data: CreditFeatures):
    if len(data.values) != len(feature_names):
        raise HTTPException(status_code=400, detail=f"Expected {len(feature_names)} features, got {len(data.values)}")

    input_df = pd.DataFrame([data.values], columns=feature_names)

    try:
        pred_class = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return {
        "predicted_class": int(pred_class),
        "risk_probability": round(float(pred_prob), 4)
    }
