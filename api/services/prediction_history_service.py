from api.repositories.patient_repository import PatientRepository
from api.repositories.prediction_repository import PredictionRepository


class PredictionHistoryService:
    def __init__(self, db):
        self.patient_repository = PatientRepository(db)
        self.prediction_repository = PredictionRepository(db)

    def get_prediction_history(self, patient_id: str):
        patient = self.patient_repository.get_by_public_id(patient_id)

        if not patient:
            return None

        return self.prediction_repository.get_by_patient_id(patient.id)