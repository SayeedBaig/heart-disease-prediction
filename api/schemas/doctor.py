from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ------------------------------------------------------------------
# Request Schemas
# ------------------------------------------------------------------

class DoctorRegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=3, max_length=100)
    specialization: str = Field(..., min_length=2, max_length=100)
    hospital: str = Field(..., min_length=2, max_length=150)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class DoctorLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class DoctorUpdateRequest(BaseModel):
    full_name: str | None = Field(None, min_length=3, max_length=100)
    specialization: str | None = Field(None, min_length=2, max_length=100)
    hospital: str | None = Field(None, min_length=2, max_length=150)


# ------------------------------------------------------------------
# Response Schemas
# ------------------------------------------------------------------

class DoctorResponse(BaseModel):
    """Public doctor profile — never exposes password_hash."""

    model_config = ConfigDict(from_attributes=True)

    doctor_id: UUID
    full_name: str
    specialization: str
    hospital: str
    email: EmailStr
    created_at: datetime


class DoctorRegisterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    doctor_id: UUID
    full_name: str
    email: EmailStr
    message: str


class DoctorLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    doctor: DoctorResponse
