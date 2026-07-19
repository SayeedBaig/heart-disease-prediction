from pydantic import BaseModel


class PublicChatRequest(BaseModel):
    question: str

from typing import Optional


class DoctorChatRequest(BaseModel):
    question: str
    risk_level: Optional[str] = None
    ecg_class: Optional[str] = None
    ef_value: Optional[float] = None