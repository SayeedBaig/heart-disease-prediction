from datetime import date, datetime, time
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.models.appointment import AppointmentStatus


# ------------------------------------------------------------------
# Request Schemas
# ------------------------------------------------------------------

class AppointmentCreate(BaseModel):
    patient_id: int
    doctor_id: UUID
    preferred_date: date
    preferred_time: time
    symptoms: str | None = Field(None, max_length=2000)
    reason: str | None = Field(None, max_length=2000)


class AppointmentUpdate(BaseModel):
    status: AppointmentStatus


# ------------------------------------------------------------------
# Response Schemas
# ------------------------------------------------------------------

class AppointmentResponse(BaseModel):
    """Full appointment record returned to clients."""

    model_config = ConfigDict(from_attributes=True)

    appointment_id: UUID
    patient_id: int
    doctor_id: UUID
    preferred_date: date
    preferred_time: time
    symptoms: str | None
    reason: str | None
    status: AppointmentStatus
    created_at: datetime
