from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.schemas.patient_request import PatientRegisterRequest
from api.schemas.patient_response import PatientRegisterResponse
from api.schemas.prediction_history_response import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
)
from api.services.patient_registration_service import PatientRegistrationService
from api.services.prediction_history_service import PredictionHistoryService


router = APIRouter(
    prefix="/patients",
    tags=["Patients"],
)


@router.post(
    "/register",
    summary="Register a new patient",
    description="Registers a new patient in the CardioAI system and returns a unique patient ID.",
    response_description="Patient registered successfully.",
    response_model=PatientRegisterResponse,
)
def register_patient(
    data: PatientRegisterRequest,
    db: Session = Depends(get_db),
):
    service = PatientRegistrationService(db)

    try:
        patient = service.register_patient(data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return PatientRegisterResponse(
        id=patient.id,
        patient_id=patient.patient_id,
        full_name=patient.full_name,
        email=patient.email,
        created_at=patient.created_at,
        message="Patient registered successfully.",
    )



@router.get(
    "/{patient_id}/predictions",
    summary="Get prediction history",
    description="Returns all past predictions for the given patient ID.",
    response_description="Prediction history retrieved successfully.",
    response_model=PredictionHistoryResponse,
)
def get_prediction_history(
    patient_id: str,
    db: Session = Depends(get_db),
):
    service = PredictionHistoryService(db)

    predictions = service.get_prediction_history(patient_id)

    if predictions is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found.",
        )

    return PredictionHistoryResponse(
        patient_id=patient_id,
        total_predictions=len(predictions),
        predictions=[
            PredictionHistoryItem(
                prediction_id=prediction.id,
                risk_level=prediction.risk_level,
                risk_percentage=prediction.risk_percentage,
                clinical_level=prediction.clinical_level,
                ecg_level=prediction.ecg_level,
                echo_level=prediction.echo_level,
                created_at=prediction.created_at,
            )
            for prediction in predictions
        ],
    )