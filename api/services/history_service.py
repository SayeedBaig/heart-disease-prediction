from fastapi import HTTPException
from sqlalchemy.orm import Session
from api.repositories.prediction_repository import PredictionRepository
from api.repositories.patient_repository import PatientRepository

class HistoryService:
    def __init__(self, db: Session):
        self.db = db
        self.prediction_repo = PredictionRepository(db)
        self.patient_repo = PatientRepository(db)

    def get_patient_history(self, patient_id: int):
        patient = self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
            
        predictions = self.prediction_repo.get_by_patient_id(patient_id)
        return {
            "patient_id": patient.id,
            "patient_code": patient.patient_id,
            "total_predictions": len(predictions),
            "predictions": predictions
        }

    def get_prediction_details(self, prediction_id: int):
        prediction = self.prediction_repo.get_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found.")
        setattr(prediction, "patient_code", prediction.patient.patient_id)
        return prediction

    def compare_predictions(self, patient_id: int):
        patient = self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")
            
        predictions = self.prediction_repo.get_by_patient_id(patient_id)
        if len(predictions) < 2:
            raise HTTPException(status_code=400, detail="Not enough predictions to compare.")
            
        # Assuming predictions are ordered by created_at desc
        latest = predictions[0]
        previous = predictions[1]
        
        return {
            "patient_id": patient_id,
            "patient_code": patient.patient_id,
            "latest": {
                "id": latest.id,
                "date": latest.created_at,
                "risk_level": latest.risk_level,
                "risk_percentage": latest.risk_percentage,
                "confidence": latest.confidence
            },
            "previous": {
                "id": previous.id,
                "date": previous.created_at,
                "risk_level": previous.risk_level,
                "risk_percentage": previous.risk_percentage,
                "confidence": previous.confidence
            },
            "trend": {
                "risk_change": latest.risk_percentage - previous.risk_percentage,
                "worsened": latest.risk_percentage > previous.risk_percentage
            }
        }
