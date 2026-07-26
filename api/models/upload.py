from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database.base import Base


class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        nullable=False,
    )

    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("predictions.id"),
        nullable=True,
    )

    file_type: Mapped[str] = mapped_column(String(20))

    original_filename: Mapped[str] = mapped_column(String(255))

    stored_filename: Mapped[str] = mapped_column(String(255))

    file_path: Mapped[str] = mapped_column(String(500))

    file_size: Mapped[int] = mapped_column(Integer)

    mime_type: Mapped[str] = mapped_column(String(100))

    status: Mapped[str] = mapped_column(
        String(20),
        default="ACTIVE",
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    patient = relationship("Patient")
    prediction = relationship("Prediction")