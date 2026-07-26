import uuid
from sqlalchemy import Column, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.database.base import Base

class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    note_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    diagnosis_id = Column(UUID(as_uuid=True), ForeignKey("diagnoses.diagnosis_id"), nullable=False)
    
    notes = Column(Text, nullable=True)
    prescription = Column(Text, nullable=True)
    advice = Column(Text, nullable=True)
    follow_up = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    diagnosis = relationship("Diagnosis", backref="doctor_note")
