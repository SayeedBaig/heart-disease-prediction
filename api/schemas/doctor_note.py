from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict

class DoctorNoteCreate(BaseModel):
    diagnosis_id: UUID
    notes: str | None = None
    prescription: str | None = None
    advice: str | None = None
    follow_up: str | None = None

class DoctorNoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    note_id: UUID
    diagnosis_id: UUID
    notes: str | None
    prescription: str | None
    advice: str | None
    follow_up: str | None
    created_at: datetime
    updated_at: datetime
