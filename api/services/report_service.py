from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.repositories.patient_repository import PatientRepository
from api.repositories.prediction_repository import PredictionRepository
from api.repositories.diagnosis_repository import DiagnosisRepository
from api.repositories.doctor_note_repository import DoctorNoteRepository
from reports.report_generator import ReportGenerator

class ReportService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self._patient_repo = PatientRepository(db)
        self._prediction_repo = PredictionRepository(db)
        self._diagnosis_repo = DiagnosisRepository(db)
        self._note_repo = DoctorNoteRepository(db)
        self._generator = ReportGenerator()

    def _resolve_patient(self, patient_id: int) -> Optional[Dict[str, Any]]:
        patient = self._patient_repo.get_by_id(patient_id)
        if patient is None:
            return None
        return {
            "patient_id": patient.patient_id,
            "full_name": patient.full_name,
            "email": patient.email,
            "phone": patient.phone,
            "gender": patient.gender,
            "date_of_birth": str(patient.date_of_birth),
        }

    def _load_prediction_data(self, prediction_id: int):
        prediction_record = self._prediction_repo.get_by_id(prediction_id)
        if not prediction_record:
            raise HTTPException(status_code=404, detail="Prediction not found.")

        # Reconstruct dicts for ReportGenerator
        prediction_dict = {
            "clinical": {"level": prediction_record.clinical_level, "score": prediction_record.clinical_score},
            "ecg": {"level": prediction_record.ecg_level, "score": prediction_record.ecg_score},
            "echo": {"level": prediction_record.echo_level, "score": prediction_record.echo_score},
            "fusion": {"final_level": prediction_record.risk_level, "risk_percentage": prediction_record.risk_percentage},
            "rag": {"explanation": prediction_record.rag_explanation}
        }
        
        explanation_dict = {"explanation": {
            "summary": prediction_record.rag_explanation,
            "details": prediction_record.rag_explanation
        }}

        digital_twin_dict = {}

        patient_dict = self._resolve_patient(prediction_record.patient_id)
        
        notes = []
        diagnosis = self._diagnosis_repo.get_by_prediction_id(prediction_id)
        if diagnosis:
            doctor_notes = self._note_repo.get_by_diagnosis_id(diagnosis.diagnosis_id)
            for n in doctor_notes:
                notes.append({
                    "notes": n.notes,
                    "prescription": n.prescription,
                    "advice": n.advice,
                    "follow_up": n.follow_up,
                    "created_at": str(n.created_at)
                })

        return prediction_dict, explanation_dict, digital_twin_dict, patient_dict, notes

    def generate_doctor_report(self, prediction_id: int) -> Dict[str, Any]:
        pred, exp, dt, patient, notes = self._load_prediction_data(prediction_id)
        report = self._generator.generate_doctor_report(pred, exp, dt, patient=patient)
        report["doctor_notes"] = notes
        return report

    def generate_patient_report(self, prediction_id: int) -> Dict[str, Any]:
        pred, exp, _, patient, notes = self._load_prediction_data(prediction_id)
        report = self._generator.generate_patient_report(pred, exp, patient=patient)
        report["doctor_notes"] = notes
        return report

