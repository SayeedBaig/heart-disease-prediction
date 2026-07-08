from datetime import date

from pydantic import BaseModel, EmailStr, Field


class PatientRegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=3, max_length=100)
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=15)
    gender: str = Field(..., pattern="^(Male|Female|Other)$")
    date_of_birth: date