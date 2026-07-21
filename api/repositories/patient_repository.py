from sqlalchemy import or_
from sqlalchemy.orm import Session

from api.models.patient import Patient


class PatientRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(self, patient_data: dict) -> Patient:
        """Add a new patient record to the session."""
        patient = Patient(**patient_data)

        self.db.add(patient)

        return patient

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, patient_id: int) -> Patient | None:
        """Return a patient by integer primary key, or None if not found."""
        return (
            self.db.query(Patient)
            .filter(Patient.id == patient_id)
            .first()
        )

    def get_by_public_id(self, patient_id: str) -> Patient | None:
        """Return a patient by public ID (e.g. PT000001), or None."""
        return (
            self.db.query(Patient)
            .filter(Patient.patient_id == patient_id)
            .first()
        )

    def get_by_email(self, email: str) -> Patient | None:
        """Return a patient by email address, or None if not found."""
        return (
            self.db.query(Patient)
            .filter(Patient.email == email)
            .first()
        )

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Patient]:
        """Return a paginated list of all patients, newest first."""
        return (
            self.db.query(Patient)
            .order_by(Patient.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_last_patient(self) -> Patient | None:
        """Return the most recently created patient (for ID generation)."""
        return (
            self.db.query(Patient)
            .order_by(Patient.id.desc())
            .first()
        )

    def search(self, query: str) -> list[Patient]:
        """Search patients by name, email, or public patient ID."""
        pattern = f"%{query}%"
        return (
            self.db.query(Patient)
            .filter(
                or_(
                    Patient.full_name.ilike(pattern),
                    Patient.email.ilike(pattern),
                    Patient.patient_id.ilike(pattern),
                )
            )
            .order_by(Patient.full_name)
            .all()
        )

    def get_or_create(self, patient_data: dict) -> Patient:
        """Return existing patient by email or create a new one."""
        patient = self.get_by_email(patient_data["email"])

        if patient:
            return patient

        return self.create(patient_data)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, patient: Patient, updates: dict) -> Patient:
        """Apply a dict of field updates to an existing patient record."""
        for field, value in updates.items():
            setattr(patient, field, value)

        return patient

    def update_password(self, patient_id: int, password_hash: str) -> Patient | None:
        """Update the password hash for a patient."""
        patient = self.get_by_id(patient_id)
        if patient:
            patient.password_hash = password_hash
        return patient


    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, patient: Patient) -> None:
        """Mark a patient record for deletion in the session."""
        self.db.delete(patient)