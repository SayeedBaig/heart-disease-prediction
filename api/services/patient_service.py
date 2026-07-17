from typing import Any, Dict

from sqlalchemy.orm import Session

from api.repositories.patient_repository import PatientRepository


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