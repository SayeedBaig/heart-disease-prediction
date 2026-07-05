from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.repositories.patient_repository import PatientRepository
from api.services.shared_memory_service import SharedMemoryService
from reports.report_generator import ReportGenerator


class ReportService:
    """
    Orchestrates report generation by combining:
      - SharedMemoryService  → prediction data
      - PatientRepository    → registered patient data
      - ReportGenerator      → report assembly + rendering

    Routes call this service and stay thin.
    ReportGenerator is not modified.
    """

    def __init__(self, db: Session) -> None:
        self._memory = SharedMemoryService()
        self._patient_repo = PatientRepository(db)
        self._generator = ReportGenerator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_memory(self) -> tuple:
        """
        Fetches the three shared-memory keys set by /predict.
        Raises 404 if no prediction has been run yet.
        """
        prediction = self._memory.get("latest_prediction")

        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail="No prediction available. Run /predict first.",
            )

        explanation = self._memory.get("latest_explanation")
        digital_twin = self._memory.get("latest_digital_twin")

        return prediction, explanation, digital_twin

    def _resolve_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Looks up the Patient ORM object by public patient_id and converts
        it to a plain dict. Returns None if patient_id is not provided.
        Raises 404 if the id is given but not found.
        """
        if not patient_id:
            return None

        patient = self._patient_repo.get_by_public_id(patient_id)

        if patient is None:
            raise HTTPException(
                status_code=404,
                detail=f"Patient '{patient_id}' not found.",
            )

        return {
            "patient_id": patient.patient_id,
            "full_name": patient.full_name,
            "email": patient.email,
            "phone": patient.phone,
            "gender": patient.gender,
            "date_of_birth": str(patient.date_of_birth),
        }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_doctor_report(
        self,
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Assembles and returns a doctor-facing report.

        Steps:
          1. Load prediction data from shared memory.
          2. Resolve patient from the database.
          3. Delegate to ReportGenerator (unchanged interface).
        """
        prediction, explanation, digital_twin = self._load_memory()

        patient = self._resolve_patient(patient_id)

        return self._generator.generate_doctor_report(
            prediction,
            explanation,
            digital_twin,
            patient=patient,
        )

    def generate_patient_report(
        self,
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Assembles and returns a patient-facing report.

        Steps:
          1. Load prediction data from shared memory.
          2. Resolve patient from the database.
          3. Delegate to ReportGenerator (unchanged interface).
        """
        prediction, explanation, _ = self._load_memory()

        patient = self._resolve_patient(patient_id)

        return self._generator.generate_patient_report(
            prediction,
            explanation,
            patient=patient,
        )
