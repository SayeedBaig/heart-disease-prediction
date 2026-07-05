from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.schemas.patient_request import PatientRegisterRequest
from api.schemas.patient_response import PatientRegisterResponse
from api.services.patient_registration_service import PatientRegistrationService

router = APIRouter(
    prefix="/patients",
    tags=["Patients"]
)


@router.post(
    "/register",
    response_model=PatientRegisterResponse
)
def register_patient(
    request: PatientRegisterRequest,
    db: Session = Depends(get_db)
):
    try:
        service = PatientRegistrationService(db)

        patient = service.register_patient(
            request.model_dump()
        )

        return {
            "id": patient.id,
            "patient_id": patient.patient_id,
            "full_name": patient.full_name,
            "email": patient.email,
            "message": "Patient registered successfully",
            "created_at": patient.created_at
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )