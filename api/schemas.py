# api/schemas.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    prices: List[float]  # last 90 close prices

class PredictionResponse(BaseModel):
    prediction: float          # predicted next-day CLOSE PRICE
    predicted_return: float    # predicted next-day % change (e.g. 0.012 = +1.2%)