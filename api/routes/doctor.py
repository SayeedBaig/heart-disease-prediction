from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.schemas.doctor import (
    DoctorLoginRequest,
    DoctorLoginResponse,
    DoctorRegisterRequest,
    DoctorRegisterResponse,
    DoctorResponse,
    DoctorUpdateRequest,
)
from api.utils.auth import get_current_doctor
from api.services.doctor_service import DoctorService


router = APIRouter(
    prefix="/doctors",
    tags=["Doctors"],
)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

@router.post(
    "/register",
    summary="Register a new doctor",
    description="Creates a new doctor account with hashed password.",
    response_description="Doctor registered successfully.",
    response_model=DoctorRegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_doctor(
    data: DoctorRegisterRequest,
    db: Session = Depends(get_db),
):
    service = DoctorService(db)

    try:
        doctor = service.register_doctor(data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return DoctorRegisterResponse(
        doctor_id=doctor.doctor_id,
        full_name=doctor.full_name,
        email=doctor.email,
        message="Doctor registered successfully.",
    )


# ------------------------------------------------------------------
# Login
# ------------------------------------------------------------------

@router.post(
    "/login",
    summary="Doctor login",
    description="Authenticates a doctor and returns a JWT access token.",
    response_description="Login successful.",
    response_model=DoctorLoginResponse,
)
def login_doctor(
    data: DoctorLoginRequest,
    db: Session = Depends(get_db),
):
    service = DoctorService(db)

    try:
        result = service.login_doctor(data.email, data.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )

    return DoctorLoginResponse(
        access_token=result["access_token"],
        doctor=DoctorResponse.model_validate(result["doctor"]),
    )


# ------------------------------------------------------------------
# Profile (Protected)
# ------------------------------------------------------------------

@router.get(
    "/me",
    summary="Get current doctor profile",
    description="Returns the profile of the authenticated doctor.",
    response_description="Doctor profile retrieved.",
    response_model=DoctorResponse,
)
def get_my_profile(
    current_doctor=Depends(get_current_doctor),
):
    return DoctorResponse.model_validate(current_doctor)


@router.put(
    "/me",
    summary="Update current doctor profile",
    description="Updates profile fields for the authenticated doctor.",
    response_description="Doctor profile updated.",
    response_model=DoctorResponse,
)
def update_my_profile(
    updates: DoctorUpdateRequest,
    current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = DoctorService(db)

    try:
        doctor = service.update_doctor_profile(
            str(current_doctor.doctor_id),
            updates.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return DoctorResponse.model_validate(doctor)


# ------------------------------------------------------------------
# Listing (Public)
# ------------------------------------------------------------------

@router.get(
    "/",
    summary="List all doctors",
    description="Returns a paginated list of registered doctors.",
    response_description="Doctors list retrieved.",
    response_model=list[DoctorResponse],
)
def list_doctors(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    service = DoctorService(db)
    doctors = service.list_doctors(skip=skip, limit=limit)

    return [DoctorResponse.model_validate(d) for d in doctors]
