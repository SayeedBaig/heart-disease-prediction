from datetime import datetime

from pydantic import BaseModel


class PredictionHistoryItem(BaseModel):
    prediction_id: int
    risk_level: str
    risk_percentage: float
    clinical_level: str
    ecg_level: str
    echo_level: str
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    patient_id: str
    total_predictions: int
    predictions: list[PredictionHistoryItem]