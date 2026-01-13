from fastapi import FastAPI, HTTPException
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict

# -------------------------
# App & Globals
# -------------------------
app = FastAPI(title="House Price Regression Inference")

MODEL_PATH = Path("models/production_model.joblib")
model = None


# -------------------------
# Startup: Load Model
# -------------------------
@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict(features: Dict[str, float]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = pd.DataFrame([features])
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
