# api/main.py
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model  # type: ignore

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
FEATURES = ["Close", "Return", "MA_10", "Volatility_10"]

# 10-day rolling features need 9 extra prior rows before the first row of
# the 90-step window has a non-NaN value, so we require at least this many
# raw prices as input.
MIN_PRICES = TIME_STEP + 9


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    prices = data.prices

    if len(prices) < MIN_PRICES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected at least {MIN_PRICES} prices "
                f"({TIME_STEP} timesteps + 9 for rolling feature "
                f"lookback), got {len(prices)}"
            )
        )

    # Build the same engineered features used in training
    df = pd.DataFrame({"Close": prices})
    df["Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["Volatility_10"] = df["Close"].rolling(window=10).std()
    df = df.dropna()

    if len(df) < TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=(
                f"After computing rolling features, only {len(df)} valid "
                f"rows remain; need at least {TIME_STEP}. Send more prices."
            )
        )

    # Take the final TIME_STEP rows (most recent window)
    window_df = df.tail(TIME_STEP)
    window = window_df[FEATURES].values
    last_close = float(window_df["Close"].iloc[-1])

    # Scale input (scaler was fit on 4 columns during training)
    window_scaled = x_scaler.transform(window)

    # Create LSTM input shape: (1, TIME_STEP, num_features)
    X = window_scaled.reshape(1, TIME_STEP, len(FEATURES))

    # Predict — the model now outputs a RETURN (e.g. 0.012 = +1.2%),
    # not a raw price, since train.py trains on target="Return".
    pred_scaled = model.predict(X)

    # Inverse scale to get the actual predicted return
    predicted_return = float(y_scaler.inverse_transform(pred_scaled)[0][0])

    # Reconstruct the predicted price from the last known close:
    # price[t] = close[t-1] * (1 + return[t])
    predicted_price = last_close * (1 + predicted_return)

    return PredictionResponse(prediction=predicted_price)