from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.repositories.patient_repository import PatientRepository
from api.schemas.request import ClinicalInput
from api.services.prediction_service import PredictionService
from api.utils.validators import validate_patient_data
from api.utils.exception_handler import handle_prediction_exception

router = APIRouter()


@router.post(
    "/predict",
    summary="Predict heart disease risk",
    description=(
        "Runs the complete CardioAI prediction pipeline using "
        "clinical data, ECG, and Echocardiography inputs."
    ),
    response_description="Prediction completed successfully.",
)
def predict(
    clinical_data: ClinicalInput,
    db: Session = Depends(get_db),
):
    try:
        patient = clinical_data.model_dump()
        ecg_path = patient.pop("ecg_path", None)
        echo_path = patient.pop("echo_path", None)
        patient_id = patient.pop("patient_id", None)

        # Resolve registered patient if patient_id was supplied
        patient_record = None
        if patient_id:
            repo = PatientRepository(db)
            patient_record = repo.get_by_patient_id(patient_id)

            if patient_record is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Patient '{patient_id}' not found.",
                )

        errors = validate_patient_data(patient)

        if errors:
            return {
                "success": False,
                "errors": errors
            }

        prediction_service = PredictionService(db)

        result = prediction_service.predict(
            clinical_data=patient,
            ecg_input=ecg_path,
            echo_input=echo_path,
            patient_record=patient_record,
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        handle_prediction_exception(e)