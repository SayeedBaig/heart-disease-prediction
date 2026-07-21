from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.schemas.patient_request import (
    PatientRegisterRequest,
    PatientLoginRequest,
)
from api.schemas.patient_response import (
    PatientRegisterResponse,
    PatientResponse,
    PatientUpdateRequest,
    TokenResponse,
    PatientLoginResponse,
    PatientMeResponse,
)
from api.schemas.prediction_history_response import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
)
from api.services.patient_registration_service import PatientRegistrationService
from api.services.patient_service import PatientService
from api.services.prediction_history_service import PredictionHistoryService
from api.utils.auth import get_current_doctor, get_current_patient


router = APIRouter(
    prefix="/patients",
    tags=["Patients"],
)


# ------------------------------------------------------------------
# Registration (Public)
# ------------------------------------------------------------------

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
@router.post(
    "/login",
    summary="Patient Login",
    description="Authenticate a patient and return a JWT access token.",
    response_model=PatientLoginResponse,
)
def login_patient(
    data: PatientLoginRequest,
    db: Session = Depends(get_db),
):
    service = PatientService(db)

    try:
        return service.login(data.email, data.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )


# ------------------------------------------------------------------
# Profile (Authenticated Patient)
# ------------------------------------------------------------------

@router.get(
    "/me",
    summary="Get authenticated patient profile",
    description="Returns the profile of the currently logged-in patient.",
    response_model=PatientMeResponse,
)
def get_me(
    current_patient = Depends(get_current_patient),
):
    return current_patient


# ------------------------------------------------------------------
# Search (must be above /{patient_id} to avoid path conflict)
# ------------------------------------------------------------------

@router.get(
    "/search",
    summary="Search patients",
    description="Search patients by name, email, or patient ID.",
    response_description="Matching patients retrieved.",
    response_model=list[PatientResponse],
)
def search_patients(
    q: str = Query(..., min_length=1, description="Search query"),
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = PatientService(db)
    patients = service.search_patients(q)

    return [PatientResponse.model_validate(p) for p in patients]


# ------------------------------------------------------------------
# List All
# ------------------------------------------------------------------

@router.get(
    "/",
    summary="List all patients",
    description="Returns a paginated list of all registered patients.",
    response_description="Patients list retrieved.",
    response_model=list[PatientResponse],
)
def list_patients(
    skip: int = 0,
    limit: int = 100,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = PatientService(db)
    patients = service.get_all_patients(skip=skip, limit=limit)

    return [PatientResponse.model_validate(p) for p in patients]


# ------------------------------------------------------------------
# Get / Update / Delete by ID
# ------------------------------------------------------------------

@router.get(
    "/{patient_id}",
    summary="Get patient by ID",
    description="Returns a single patient by their integer primary key.",
    response_description="Patient retrieved.",
    response_model=PatientResponse,
)
def get_patient(
    patient_id: int,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = PatientService(db)

    try:
        patient = service.get_patient(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return PatientResponse.model_validate(patient)


@router.put(
    "/{patient_id}",
    summary="Update patient",
    description="Updates profile fields for an existing patient.",
    response_description="Patient updated.",
    response_model=PatientResponse,
)
def update_patient(
    patient_id: int,
    updates: PatientUpdateRequest,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = PatientService(db)

    try:
        patient = service.update_patient(patient_id, updates.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return PatientResponse.model_validate(patient)


@router.delete(
    "/{patient_id}",
    summary="Delete patient",
    description="Permanently removes a patient record.",
    response_description="Patient deleted.",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_patient(
    patient_id: int,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = PatientService(db)

    try:
        service.delete_patient(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ------------------------------------------------------------------
# Prediction History
# ------------------------------------------------------------------

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
    current_patient = Depends(get_current_patient),
):
    if patient_id != current_patient.patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )
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


# Singular APIRouter for /patient/login compatibility
singular_router = APIRouter(tags=["Patients"])


@singular_router.post(
    "/patient/login",
    summary="Patient Login (Singular)",
    description="Authenticate a patient and return a JWT access token.",
    response_model=PatientLoginResponse,
)
def login_patient_singular(
    data: PatientLoginRequest,
    db: Session = Depends(get_db),
):
    return login_patient(data, db)