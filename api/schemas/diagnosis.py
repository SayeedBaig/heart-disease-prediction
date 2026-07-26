from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class ClinicalDataSchema(BaseModel):
    gender: int = Field(..., ge=1, le=2)
    height: float = Field(..., ge=100, le=250)
    weight: float = Field(..., ge=30, le=300)
    ap_hi: int = Field(..., ge=50, le=300)
    ap_lo: int = Field(..., ge=30, le=200)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)

class DiagnosisCreate(BaseModel):
    appointment_id: UUID
    clinical_data: ClinicalDataSchema
    ecg_path: str | None = None
    echo_path: str | None = None


class DiagnosisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    diagnosis_id: UUID
    appointment_id: UUID
    patient_id: int
    doctor_id: UUID
    clinical_data: dict
    ecg_path: str | None
    echo_path: str | None
    status: str
    prediction_id: int | None
    created_at: datetime
