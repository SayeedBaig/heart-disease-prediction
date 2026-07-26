from datetime import date
from uuid import UUID
from sqlalchemy.orm import Session

from api.models.appointment import AppointmentStatus
from api.models.diagnosis import DiagnosisStatus
from api.repositories.diagnosis_repository import DiagnosisRepository
from api.repositories.appointment_repository import AppointmentRepository
from api.repositories.patient_repository import PatientRepository
from api.services.prediction_service import PredictionService

class DiagnosisService:
    def __init__(self, db: Session):
        self.db = db
        self.diagnosis_repo = DiagnosisRepository(db)
        self.appointment_repo = AppointmentRepository(db)
        self.patient_repo = PatientRepository(db)

    def create_diagnosis(self, diagnosis_data: dict, doctor_id: UUID):
        # Validate appointment
        appointment = self.appointment_repo.get_by_id(diagnosis_data["appointment_id"])
        if not appointment:
            raise ValueError("Appointment not found.")
        
        # We need doctor_id constraint? Actually yes, ensure it's their appointment
        # Wait, appointment uses string for doctor_id internally or UUID?
        # doctor_id in Appointment is UUID.
        if str(appointment.doctor_id) != str(doctor_id):
            raise ValueError("Unauthorized to create diagnosis for this appointment.")

        # Validate patient
        patient = self.patient_repo.get_by_id(appointment.patient_id)
        if not patient:
            raise ValueError("Patient not found.")

        # Calculate age
        today = date.today()
        dob = patient.date_of_birth
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        clinical_data = diagnosis_data["clinical_data"].copy()
        clinical_data["age"] = age

        new_data = {
            "appointment_id": appointment.appointment_id,
            "patient_id": patient.id,
            "doctor_id": doctor_id,
            "clinical_data": clinical_data,
            "ecg_path": diagnosis_data.get("ecg_path"),
            "echo_path": diagnosis_data.get("echo_path"),
            "status": DiagnosisStatus.PENDING,
        }

        try:
            diagnosis = self.diagnosis_repo.create(new_data)
            self.db.commit()
            self.db.refresh(diagnosis)
        except Exception:
            self.db.rollback()
            raise

        return diagnosis

    def get_diagnosis(self, diagnosis_id: UUID, doctor_id: UUID):
        diagnosis = self.diagnosis_repo.get_by_id(diagnosis_id)
        if not diagnosis:
            raise ValueError("Diagnosis not found.")
        if str(diagnosis.doctor_id) != str(doctor_id):
            raise ValueError("Unauthorized access to this diagnosis.")
        return diagnosis

    def run_prediction(self, diagnosis_id: UUID, doctor_id: UUID):
        diagnosis = self.get_diagnosis(diagnosis_id, doctor_id)
        
        if diagnosis.status == DiagnosisStatus.COMPLETED:
            raise ValueError("Prediction already completed for this diagnosis.")
            
        try:
            self.diagnosis_repo.update_status(diagnosis, DiagnosisStatus.PREDICTING)
            self.db.commit()
            self.db.refresh(diagnosis)
        except Exception:
            self.db.rollback()
            raise

        # Prepare for prediction
        patient_record = self.patient_repo.get_by_id(diagnosis.patient_id)
        prediction_service = PredictionService(self.db)
        
        try:
            result = prediction_service.predict(
                clinical_data=diagnosis.clinical_data,
                ecg_input=diagnosis.ecg_path,
                echo_input=diagnosis.echo_path,
                patient_record=patient_record
            )
            
            if not result.get("success", True): # handle validation errors
                self.diagnosis_repo.update_status(diagnosis, DiagnosisStatus.FAILED)
                self.db.commit()
                raise ValueError(f"Prediction failed: {result.get('errors')}")

            # Get the saved prediction from the DB (the last one saved for this patient)
            # Or we can get it from the result if we modify prediction_service to return prediction_id.
            # PredictionService saves to DB but doesn't return the ID in the response directly.
            # Let's fetch the most recent prediction for this patient.
            from api.repositories.prediction_repository import PredictionRepository
            pred_repo = PredictionRepository(self.db)
            recent_preds = pred_repo.get_by_patient_id(patient_record.id)
            if not recent_preds:
                raise ValueError("Prediction was not saved properly.")
                
            latest_prediction = recent_preds[0]
            
            self.diagnosis_repo.set_prediction(diagnosis, latest_prediction.id)
            appointment = self.appointment_repo.get_by_id(diagnosis.appointment_id)
            if appointment and appointment.status != AppointmentStatus.COMPLETED:
                self.appointment_repo.update_status(
                    appointment, AppointmentStatus.COMPLETED
                )
            self.db.commit()
            self.db.refresh(diagnosis)
            
            # Standardize response as requested
            return {
                "prediction_id": latest_prediction.id,
                "diagnosis_id": diagnosis.diagnosis_id,
                "risk_level": latest_prediction.risk_level,
                "risk_percentage": latest_prediction.risk_percentage,
                "confidence": latest_prediction.confidence,
                "prediction_timestamp": latest_prediction.created_at,
                "rag_explanation": latest_prediction.rag_explanation,
                "raw_result": result
            }
        except Exception as e:
            self.db.rollback()
            self.diagnosis_repo.update_status(diagnosis, DiagnosisStatus.FAILED)
            self.db.commit()
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Prediction pipeline error: {str(e)}")
