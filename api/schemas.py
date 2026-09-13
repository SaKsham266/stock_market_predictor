# api/schemas.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    # Raw close prices only — the API derives Return, MA_10, and
    # Volatility_10 from these server-side (see api/main.py).
    # Must include at least 99 prices: 90 model timesteps + 9 extra
    # days of lookback, since MA_10/Volatility_10 need a 10-day window
    # before the first row of the 90-step sequence is non-NaN.
    prices: List[float]

class PredictionResponse(BaseModel):
    prediction: float