from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class AgentResult(BaseModel):
    level: Optional[str]
    score: float
    reason: str


class FusionResult(BaseModel):
    final_level: str
    risk_percentage: float


class RagResult(BaseModel):
    explanation: str
    details: List[str]


class PredictResponse(BaseModel):
    clinical: AgentResult
    ecg: AgentResult
    echo: AgentResult
    fusion: FusionResult
    rag: RagResult


class PredictEndpointResponse(BaseModel):
    prediction: PredictResponse
    explanation: Dict[str, Any]
    digital_twin: Dict[str, Any]
    prediction_id: Optional[str] = None