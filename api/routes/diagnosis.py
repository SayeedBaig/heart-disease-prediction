from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.schemas.diagnosis import DiagnosisCreate, DiagnosisResponse
from api.services.diagnosis_service import DiagnosisService
from api.utils.auth import get_current_doctor
from api.repositories.prediction_repository import PredictionRepository


router = APIRouter()


class PredictionRequest(BaseModel):
    diagnosis_id: UUID


@router.post(
    "/diagnosis",
    summary="Create a new diagnosis",
    description="Creates a diagnosis record associated with an appointment.",
    response_model=DiagnosisResponse,
    tags=["Diagnosis"]
)
def create_diagnosis(
    data: DiagnosisCreate,
    db: Session = Depends(get_db),
    current_doctor=Depends(get_current_doctor)
):
    service = DiagnosisService(db)
    try:
        diagnosis = service.create_diagnosis(data.model_dump(), current_doctor.doctor_id)
        return DiagnosisResponse.model_validate(diagnosis)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/diagnosis/{diagnosis_id}",
    summary="Get diagnosis details",
    description="Fetches a diagnosis by ID.",
    response_model=DiagnosisResponse,
    tags=["Diagnosis"]
)
def get_diagnosis(
    diagnosis_id: UUID,
    db: Session = Depends(get_db),
    current_doctor=Depends(get_current_doctor)
):
    service = DiagnosisService(db)
    try:
        diagnosis = service.get_diagnosis(diagnosis_id, current_doctor.doctor_id)
        return DiagnosisResponse.model_validate(diagnosis)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post(
    "/prediction",
    summary="Run prediction on a diagnosis",
    description="Invokes the prediction pipeline for a given diagnosis.",
    tags=["Prediction"]
)
def run_prediction(
    data: PredictionRequest,
    db: Session = Depends(get_db),
    current_doctor=Depends(get_current_doctor)
):
    service = DiagnosisService(db)
    try:
        result = service.run_prediction(data.diagnosis_id, current_doctor.doctor_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/prediction/{prediction_id}",
    summary="Get prediction details",
    description="Fetches standard prediction results by prediction ID.",
    tags=["Prediction"]
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_doctor=Depends(get_current_doctor)
):
    # Retrieve the prediction using PredictionRepository
    repo = PredictionRepository(db)
    prediction = repo.get_by_id(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found.")
        
    return {
        "prediction_id": prediction.id,
        "patient_id": prediction.patient_id,
        "risk_level": prediction.risk_level,
        "risk_percentage": prediction.risk_percentage,
        "confidence": prediction.confidence,
        "prediction_timestamp": prediction.created_at,
        "rag_explanation": prediction.rag_explanation
    }
