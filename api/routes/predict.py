from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.repositories.patient_repository import PatientRepository
from api.schemas.request import ClinicalInput
from api.schemas.response import PredictEndpointResponse
from api.services.prediction_service import PredictionService
from api.utils.validators import validate_patient_data
from api.utils.exception_handler import handle_prediction_exception
from api.models.diagnosis import DiagnosisStatus

router = APIRouter()


@router.post(
    "/predict",
    summary="Predict heart disease risk",
    description=(
        "Runs the complete CardioAI prediction pipeline using "
        "clinical data, ECG, and Echocardiography inputs."
    ),
    response_description="Prediction completed successfully.",
    responses={200: {"model": PredictEndpointResponse}},
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
            patient_record = repo.get_by_public_id(patient_id)
            
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
        print("Patient ID received:", patient_id)
        print("Patient Record:", patient_record)
        print("Diagnoses:", patient_record.diagnoses)
        print("Diagnosis Count:", len(patient_record.diagnoses))

        for d in patient_record.diagnoses:
            print("Diagnosis:", d.diagnosis_id, d.status)

        prediction_service = PredictionService(db)
        

        result = prediction_service.predict(
            clinical_data=patient,
            ecg_input=ecg_path,
            echo_input=echo_path,
            patient_record=patient_record,
        )

        result["diagnosis_id"] = None
        if patient_record and patient_record.diagnoses:
            pending_diagnoses = [
                d for d in patient_record.diagnoses 
                if d.status == DiagnosisStatus.PENDING
            ]
            if pending_diagnoses:
                latest_diagnosis = sorted(pending_diagnoses, key=lambda d: d.created_at)[-1]
                latest_diagnosis.prediction_id = result.get("prediction_id")
                latest_diagnosis.status = DiagnosisStatus.COMPLETED
                db.commit()
                result["diagnosis_id"] = latest_diagnosis.diagnosis_id

        return result

    except HTTPException:
        raise

    except Exception as e:
        handle_prediction_exception(e)