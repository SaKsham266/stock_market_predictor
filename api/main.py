# api/main.py
from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore

from api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Stock Price Predictor API")

# Load model & scalers ONCE at startup
MODEL_PATH = "model/saved_model/lstm_model.keras"
X_SCALER_PATH = "model/saved_model/x_scaler.pkl"
Y_SCALER_PATH = "model/saved_model/y_scaler.pkl"

model = load_model(MODEL_PATH)
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

TIME_STEP = 90


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    prices = data.prices

    if len(prices) != TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {TIME_STEP} prices, got {len(prices)}"
        )

    # Convert to numpy
    prices = np.array(prices).reshape(-1, 1)

    # Scale input
    prices_scaled = x_scaler.transform(prices)

    # Create LSTM input shape: (1, TIME_STEP, 1)
    X = prices_scaled.reshape(1, TIME_STEP, 1)

    # Predict
    pred_scaled = model.predict(X)

    # Inverse scale
    pred = y_scaler.inverse_transform(pred_scaled)

    return PredictionResponse(prediction=float(pred[0][0]))
