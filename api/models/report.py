from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database.base import Base


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        nullable=False,
    )

    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id"),
        nullable=False,
    )

    report_type: Mapped[str] = mapped_column(
        String(20)
    )

    report_format: Mapped[str] = mapped_column(
        String(20)
    )

    generated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    patient = relationship("Patient")
    prediction = relationship("Prediction")