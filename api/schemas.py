# api/schemas.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    prices: List[float]  # last 90 close prices

class PredictionResponse(BaseModel):
    prediction: float
