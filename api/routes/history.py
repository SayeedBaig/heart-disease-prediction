from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.database.session import get_db
from api.services.history_service import HistoryService
from api.utils.auth import get_current_doctor
from api.schemas.history_response import (
    PatientHistoryResponse,
    CompareHistoryResponse,
    PredictionDetailResponse,
)

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/patient/{patient_id}", summary="Get patient prediction history", response_model=PatientHistoryResponse)
def get_patient_history(patient_id: int, db: Session = Depends(get_db), current_doctor=Depends(get_current_doctor)):
    return HistoryService(db).get_patient_history(patient_id)

@router.get("/patient/{patient_id}/compare", summary="Compare latest vs previous prediction", response_model=CompareHistoryResponse)
def compare_predictions(patient_id: int, db: Session = Depends(get_db), current_doctor=Depends(get_current_doctor)):
    return HistoryService(db).compare_predictions(patient_id)

@router.get("/prediction/{prediction_id}", summary="Get prediction details", response_model=PredictionDetailResponse)
def get_prediction_details(prediction_id: int, db: Session = Depends(get_db), current_doctor=Depends(get_current_doctor)):
    return HistoryService(db).get_prediction_details(prediction_id)
