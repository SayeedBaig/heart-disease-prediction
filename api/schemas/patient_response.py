from datetime import datetime

from pydantic import BaseModel, EmailStr


class PatientRegisterResponse(BaseModel):
    id: int
    patient_id: str
    full_name: str
    email: EmailStr
    message: str
    created_at: datetime