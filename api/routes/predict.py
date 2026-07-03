from fastapi import APIRouter

from api.schemas.request import ClinicalInput
from api.services.prediction_service import PredictionService
from api.utils.validators import validate_patient_data
from api.utils.exception_handler import handle_prediction_exception

router = APIRouter()

prediction_service = PredictionService()


@router.post("/predict")
def predict(clinical_data: ClinicalInput):
    try:
        patient = clinical_data.model_dump()
        ecg_path = patient.pop("ecg_path", None)
        echo_path = patient.pop("echo_path", None)

        errors = validate_patient_data(patient)

        if errors:
            return {
                "success": False,
                "errors": errors
            }

        result = prediction_service.predict(
            clinical_data=patient,
            ecg_input=ecg_path,
            echo_input=echo_path,
        )

        return result

    except Exception as e:
        handle_prediction_exception(e)