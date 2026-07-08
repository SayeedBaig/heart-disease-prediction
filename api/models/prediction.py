from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        nullable=False
    )

    clinical_level: Mapped[str] = mapped_column(String(10))
    clinical_score: Mapped[float] = mapped_column(Float)

    ecg_level: Mapped[str] = mapped_column(String(10))
    ecg_score: Mapped[float] = mapped_column(Float)

    echo_level: Mapped[str] = mapped_column(String(10))
    echo_score: Mapped[float] = mapped_column(Float)

    risk_level: Mapped[str] = mapped_column(String(10))
    risk_percentage: Mapped[float] = mapped_column(Float)

    rag_explanation: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    patient = relationship("Patient", back_populates="predictions")