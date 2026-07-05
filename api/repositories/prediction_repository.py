from sqlalchemy.orm import Session

from api.models.prediction import Prediction


class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def save(self, patient_db_id: int, prediction: dict):
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
            rag_explanation=prediction["prediction"]["rag"]["explanation"],
        )
    def get_by_patient_id(self, patient_db_id: int):
        return (
            self.db.query(Prediction)
            .filter(Prediction.patient_id == patient_db_id)
            .order_by(Prediction.created_at.desc())
            .all()
        )


    def get_by_id(self, prediction_id: int):
        return (
                self.db.query(Prediction)
                .filter(Prediction.id == prediction_id)
                .first()
        )   

        self.db.add(prediction_db)
        self.db.commit()
        self.db.refresh(prediction_db)

        return prediction_db