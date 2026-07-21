from typing import Any, Dict

from sqlalchemy.orm import Session

from api.repositories.patient_repository import PatientRepository
from api.utils.auth import create_access_token, verify_password
from api.utils.security import hash_password


class PatientService:
    """
    Handles patient CRUD operations and data preparation.

    The prepare_patient method is used by the prediction pipeline
    to standardize patient data before inference.
    """

    def __init__(self, db: Session | None = None):
        self.db = db
        self.patient_repo = PatientRepository(db) if db else None

    # ------------------------------------------------------------------
    # Data Preparation (used by prediction pipeline)
    # ------------------------------------------------------------------

    def prepare_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and standardize patient information."""
        patient = patient_data.copy()

        return patient

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_patient(self, patient_id: int):
        """Return a single patient by integer PK or raise if not found."""
        patient = self.patient_repo.get_by_id(patient_id)

        if not patient:
            raise ValueError("Patient not found.")

        return patient

    def get_all_patients(self, skip: int = 0, limit: int = 100):
        """Return a paginated list of all patients."""
        return self.patient_repo.get_all(skip=skip, limit=limit)

    def search_patients(self, query: str):
        """Search patients by name, email, or public patient ID."""
        return self.patient_repo.search(query)
    def login(self, email: str, password: str):
        """Authenticate a patient and return a JWT token."""
        patient = self.patient_repo.get_by_email(email)

        if not patient:
            raise ValueError("Invalid email or password.")

        if not verify_password(password, patient.password_hash):
            raise ValueError("Invalid email or password.")

        token = create_access_token(
            {
                "sub": str(patient.id),
                "role": "patient",
            }
        )

        return {
            "access_token": token,
            "token_type": "bearer",
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_patient(self, patient_id: int, updates: dict):
        """Update and return the patient profile.

        Raises ValueError if the patient is not found.
        """
        patient = self.patient_repo.get_by_id(patient_id)

        if not patient:
            raise ValueError("Patient not found.")

        clean_updates = {k: v for k, v in updates.items() if v is not None}

        if not clean_updates:
            return patient

        try:
            patient = self.patient_repo.update(patient, clean_updates)
            self.db.commit()
            self.db.refresh(patient)
        except Exception:
            self.db.rollback()
            raise

        return patient

    def set_password(self, patient_id: int, plain_password: str):
        """Hash and set the patient's password."""
        patient = self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise ValueError("Patient not found.")

        hashed = hash_password(plain_password)
        try:
            self.patient_repo.update_password(patient.id, hashed)
            self.db.commit()
            self.db.refresh(patient)
        except Exception:
            self.db.rollback()
            raise
        return patient

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_patient(self, patient_id: int):
        """Delete a patient by integer PK.  Raises ValueError if not found."""
        patient = self.patient_repo.get_by_id(patient_id)

        if not patient:
            raise ValueError("Patient not found.")

        try:
            self.patient_repo.delete(patient)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise