from sqlalchemy.orm import Session

from api.models.doctor import Doctor


class DoctorRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_doctor(self, doctor_data: dict) -> Doctor:
        """Add a new doctor record to the session."""
        doctor = Doctor(**doctor_data)

        self.db.add(doctor)

        return doctor

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_doctor_by_id(self, doctor_id: str) -> Doctor | None:
        """Return a doctor by UUID primary key, or None if not found."""
        return (
            self.db.query(Doctor)
            .filter(Doctor.doctor_id == doctor_id)
            .first()
        )

    def get_doctor_by_email(self, email: str) -> Doctor | None:
        """Return a doctor by email address, or None if not found."""
        return (
            self.db.query(Doctor)
            .filter(Doctor.email == email)
            .first()
        )

    def list_doctors(self, skip: int = 0, limit: int = 100) -> list[Doctor]:
        """Return a paginated list of all doctors, ordered by name."""
        return (
            self.db.query(Doctor)
            .order_by(Doctor.full_name)
            .offset(skip)
            .limit(limit)
            .all()
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_doctor(self, doctor: Doctor, updates: dict) -> Doctor:
        """Apply a dict of field updates to an existing doctor record."""
        for field, value in updates.items():
            setattr(doctor, field, value)

        return doctor

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_doctor(self, doctor: Doctor) -> None:
        """Mark a doctor record for deletion in the session."""
        self.db.delete(doctor)
