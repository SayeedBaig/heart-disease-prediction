from fastapi import APIRouter

from api.schemas.request import ClinicalInput
from api.services.prediction_service import PredictionService

router = APIRouter()

prediction_service = PredictionService()


@router.post("/predict")
def predict(clinical_data: ClinicalInput):
    result = prediction_service.predict(
        clinical_data=clinical_data.model_dump(),
        ecg_input="sample_ecg.csv",
        echo_input=None,
    )

    return result