from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class HistoryPredictionItem(BaseModel):
    id: int
    patient_id: int
    clinical_level: str
    clinical_score: float
    ecg_level: str
    ecg_score: float
    echo_level: str
    echo_score: float
    risk_level: str
    risk_percentage: float
    confidence: Optional[float]
    rag_explanation: str
    created_at: datetime

    class Config:
        from_attributes = True


class PatientHistoryResponse(BaseModel):
    patient_id: int
    patient_code: str
    total_predictions: int
    predictions: List[HistoryPredictionItem]


class ComparePredictionItem(BaseModel):
    id: int
    date: datetime
    risk_level: str
    risk_percentage: float
    confidence: Optional[float]


class TrendInfo(BaseModel):
    risk_change: float
    worsened: bool


class CompareHistoryResponse(BaseModel):
    patient_id: int
    patient_code: str
    latest: ComparePredictionItem
    previous: ComparePredictionItem
    trend: TrendInfo


class PredictionDetailResponse(HistoryPredictionItem):
    patient_code: str
