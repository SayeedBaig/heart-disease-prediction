import enum
import uuid

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from api.database.base import Base


class DiagnosisStatus(str, enum.Enum):
    PENDING = "Pending"
    PREDICTING = "Predicting"
    COMPLETED = "Completed"
    FAILED = "Failed"


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    diagnosis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    appointment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("appointments.appointment_id"),
        nullable=False
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False
    )

    doctor_id = Column(
        UUID(as_uuid=True),
        ForeignKey("doctors.doctor_id"),
        nullable=False
    )

    clinical_data = Column(JSONB, nullable=False)
    ecg_path = Column(String(500), nullable=True)
    echo_path = Column(String(500), nullable=True)

    status = Column(
        Enum(
            DiagnosisStatus,
            name="diagnosisstatus",
            values_callable=lambda obj: [e.value for e in obj]
        ),
        nullable=False,
        default=DiagnosisStatus.PENDING,
        server_default=DiagnosisStatus.PENDING.value
    )

    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    appointment = relationship("Appointment", backref="diagnoses")
    patient = relationship("Patient", backref="diagnoses")
    doctor = relationship("Doctor", backref="diagnoses")
    prediction = relationship("Prediction", backref="diagnosis")
