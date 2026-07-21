from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class PatientRegisterResponse(BaseModel):
    """
    Response returned after a successful patient registration.
    Reads directly from the SQLAlchemy Patient ORM object via from_attributes.
    The 'message' field is injected by the route layer since it is not on the ORM model.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: str
    full_name: str
    email: EmailStr
    message: str
    created_at: datetime


class PatientResponse(BaseModel):
    """Full patient record returned to clients."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: str
    full_name: str
    email: EmailStr
    phone: str
    gender: str
    date_of_birth: date
    created_at: datetime
    updated_at: datetime


class PatientUpdateRequest(BaseModel):
    full_name: str | None = Field(None, min_length=3, max_length=100)
    phone: str | None = Field(None, min_length=10, max_length=15)
    gender: str | None = Field(None, pattern="^(Male|Female|Other)$")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PatientProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    patient_id: str
    full_name: str
    email: EmailStr
    phone: str
    gender: str
    date_of_birth: date


class PatientLoginResponsePatient(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: str
    full_name: str
    email: EmailStr


class PatientLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    patient: PatientLoginResponsePatient


class PatientMeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: str
    full_name: str
    email: EmailStr
    phone: str
    age: int
    gender: str
