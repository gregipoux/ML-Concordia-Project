"""
FastAPI REST API for Cybersecurity Intrusion Detection.
Serves predictions from the trained model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Cybersecurity Intrusion Detection API",
    description="Predicts whether a network session is a cyber attack or normal traffic.",
    version="1.0.0",
)


# --- Pydantic models ---

class SessionInput(BaseModel):
    """Input schema matching dataset features."""
    network_packet_size: int = Field(..., ge=64, le=1500, description="Packet size in bytes")
    protocol_type: str = Field(..., description="TCP, UDP, or ICMP")
    login_attempts: int = Field(..., ge=0, description="Number of login attempts")
    session_duration: float = Field(..., ge=0, description="Session length in seconds")
    encryption_used: str = Field(..., description="AES, DES, or None")
    ip_reputation_score: float = Field(..., ge=0, le=1, description="IP reputation (0=clean, 1=suspicious)")
    failed_logins: int = Field(..., ge=0, description="Number of failed login attempts")
    browser_type: str = Field(..., description="Chrome, Firefox, Edge, Safari, or Unknown")
    unusual_time_access: int = Field(..., ge=0, le=1, description="1 if accessed at unusual time")

    class Config:
        json_schema_extra = {
            "example": {
                "network_packet_size": 599,
                "protocol_type": "TCP",
                "login_attempts": 4,
                "session_duration": 492.98,
                "encryption_used": "DES",
                "ip_reputation_score": 0.61,
                "failed_logins": 1,
                "browser_type": "Edge",
                "unusual_time_access": 0,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    risk_level: str


# --- Model loading ---

MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/best_model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "saved_models/preprocessor.joblib")

model = None
preprocessor = None
model_metadata: dict = {}


@app.on_event("startup")
async def load_model():
    """
    Load champion artifact. Supports two formats:
    - Bundle dict saved by notebook 05: {'model', 'preprocessor', 'champion_name', ...}
    - Legacy: separate `best_model.joblib` + `preprocessor.joblib`
    """
    global model, preprocessor, model_metadata
    try:
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and "model" in bundle and "preprocessor" in bundle:
            model = bundle["model"]
            preprocessor = bundle["preprocessor"]
            model_metadata = {k: v for k, v in bundle.items() if k not in ("model", "preprocessor")}
        else:
            model = bundle
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            model_metadata = {}
        print(f"Model loaded from {MODEL_PATH} ({type(model).__name__})")
    except FileNotFoundError as e:
        print(f"WARNING: Model not found ({e}). Train the model first. API will return errors on /predict.")


# --- Endpoints ---

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH,
        "features_expected": 9,
        "champion_name": model_metadata.get("champion_name"),
        "default_threshold": model_metadata.get("default_threshold", 0.5),
        "training_metadata": model_metadata.get("metadata"),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(session: SessionInput):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded. Train first.")

    # Convert to DataFrame for preprocessing
    input_df = pd.DataFrame([session.model_dump()])

    # Apply feature engineering
    from src.utils.feature_engineering import add_all_features
    input_df = add_all_features(input_df)

    # Preprocess
    X = preprocessor.transform(input_df)

    # Predict
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(prediction)

    # Risk level
    if probability > 0.8:
        risk_level = "CRITICAL"
    elif probability > 0.5:
        risk_level = "HIGH"
    elif probability > 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        label="Attack Detected" if prediction == 1 else "Normal Traffic",
        risk_level=risk_level,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
