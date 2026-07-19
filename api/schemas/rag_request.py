from pydantic import BaseModel


class PublicChatRequest(BaseModel):
    question: str

from typing import Optional


class DoctorChatRequest(BaseModel):
    question: str
    risk_level: Optional[str] = None
    ecg_class: Optional[str] = None
    ef_value: Optional[float] = None


class FoodRecommendationRequest(BaseModel):
    risk_level: str
    diabetes: Optional[bool] = False
    smoker: Optional[bool] = False
    age: Optional[int] = None


class PatientChatRequest(BaseModel):
    question: str
    risk_level: Optional[str] = None