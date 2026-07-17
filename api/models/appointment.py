import enum
import uuid

from sqlalchemy import Column, Date, DateTime, Enum, ForeignKey, Text, Time
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from api.database.base import Base


class AppointmentStatus(str, enum.Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    COMPLETED = "Completed"


class Appointment(Base):
    __tablename__ = "appointments"

    appointment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    patient_id = Column(
        ForeignKey("patients.id"),
        nullable=False
    )

    doctor_id = Column(
        UUID(as_uuid=True),
        ForeignKey("doctors.doctor_id"),
        nullable=False
    )

    preferred_date = Column(Date, nullable=False)
    preferred_time = Column(Time, nullable=False)
    symptoms = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)

    status = Column(
        Enum(
            AppointmentStatus,
            name="appointmentstatus",
            values_callable=lambda obj: [e.value for e in obj]
        ),
        nullable=False,
        default=AppointmentStatus.PENDING,
        server_default=AppointmentStatus.PENDING.value
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # ── Relationships ──────────────────────────────────────────────────
    patient = relationship("Patient", backref="appointments")
    doctor = relationship("Doctor", backref="appointments")
