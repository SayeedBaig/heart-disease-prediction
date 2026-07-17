from uuid import UUID
from sqlalchemy.orm import Session
from api.models.diagnosis import Diagnosis, DiagnosisStatus

class DiagnosisRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, diagnosis_data: dict) -> Diagnosis:
        diagnosis = Diagnosis(**diagnosis_data)
        self.db.add(diagnosis)
        return diagnosis

    def get_by_id(self, diagnosis_id: UUID) -> Diagnosis | None:
        return self.db.query(Diagnosis).filter(Diagnosis.diagnosis_id == diagnosis_id).first()

    def update_status(self, diagnosis: Diagnosis, status: DiagnosisStatus) -> Diagnosis:
        diagnosis.status = status
        return diagnosis

    def set_prediction(self, diagnosis: Diagnosis, prediction_id: int) -> Diagnosis:
        diagnosis.prediction_id = prediction_id
        diagnosis.status = DiagnosisStatus.COMPLETED
        return diagnosis

    def get_by_prediction_id(self, prediction_id: int) -> Diagnosis | None:
        return self.db.query(Diagnosis).filter(Diagnosis.prediction_id == prediction_id).first()
