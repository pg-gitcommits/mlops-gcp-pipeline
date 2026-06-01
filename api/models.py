from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    """Response schema for a single image prediction."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model: str
    checkpoint: str