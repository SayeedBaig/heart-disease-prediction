from sqlalchemy.orm import Session

from api.models.prediction import Prediction


class PredictionRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    def save(self, patient_db_id: int, prediction: dict) -> Prediction:
        """Persist a new prediction record and return it."""
        prediction_db = Prediction(
            patient_id=patient_db_id,
            clinical_level=prediction["prediction"]["clinical"]["level"],
            clinical_score=prediction["prediction"]["clinical"]["score"],
            ecg_level=prediction["prediction"]["ecg"]["level"],
            ecg_score=prediction["prediction"]["ecg"]["score"],
            echo_level=prediction["prediction"]["echo"]["level"],
            echo_score=prediction["prediction"]["echo"]["score"],
            risk_level=prediction["prediction"]["fusion"]["final_level"],
            risk_percentage=prediction["prediction"]["fusion"]["risk_percentage"],
            confidence=prediction["prediction"]["fusion"].get("risk_percentage", 0) / 100.0, # Confidence can be derived from risk percentage if fusion model doesn't output it directly, or maybe clinical/ecg score. Let's just use risk_percentage / 100
            rag_explanation=prediction["prediction"]["rag"]["explanation"],
        )

        self.db.add(prediction_db)
        self.db.commit()
        self.db.refresh(prediction_db)

        return prediction_db

    def get_by_patient_id(self, patient_db_id: int) -> list:
        """Return all predictions for a patient, newest first."""
        return (
            self.db.query(Prediction)
            .filter(Prediction.patient_id == patient_db_id)
            .order_by(Prediction.created_at.desc())
            .all()
        )

    def get_by_id(self, prediction_id: int):
        """Return a single prediction by its primary key."""
        return (
            self.db.query(Prediction)
            .filter(Prediction.id == prediction_id)
            .first()
        )