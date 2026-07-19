from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.schemas.appointment import (
    AppointmentCreate,
    AppointmentResponse,
)
from api.services.appointment_service import AppointmentService
from api.utils.auth import get_current_doctor


router = APIRouter(
    prefix="/appointments",
    tags=["Appointments"],
)


# ------------------------------------------------------------------
# Create
# ------------------------------------------------------------------

@router.post(
    "",
    summary="Book a new appointment",
    description="Creates a new appointment record for a patient with a doctor.",
    response_description="Appointment created successfully.",
    response_model=AppointmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_appointment(
    data: AppointmentCreate,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)

    try:
        appointment = service.create_appointment(data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return AppointmentResponse.model_validate(appointment)


# ------------------------------------------------------------------
# Read — Single
# ------------------------------------------------------------------

@router.get(
    "/{appointment_id}",
    summary="Get appointment by ID",
    description="Returns a single appointment by its UUID.",
    response_description="Appointment retrieved.",
    response_model=AppointmentResponse,
)
def get_appointment(
    appointment_id: str,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)

    try:
        appointment = service.get_appointment(appointment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return AppointmentResponse.model_validate(appointment)


# ------------------------------------------------------------------
# Read — Lists
# ------------------------------------------------------------------

@router.get(
    "",
    summary="List all appointments",
    description="Returns a paginated list of all appointments.",
    response_description="Appointments list retrieved.",
    response_model=list[AppointmentResponse],
)
def get_all_appointments(
    skip: int = 0,
    limit: int = 100,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)
    appointments = service.get_all_appointments(skip=skip, limit=limit)

    return [AppointmentResponse.model_validate(a) for a in appointments]


@router.get(
    "/patient/{patient_id}",
    summary="Get appointments by patient",
    description="Returns all appointments for a specific patient.",
    response_description="Patient appointments retrieved.",
    response_model=list[AppointmentResponse],
)
def get_patient_appointments(
    patient_id: int,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)
    appointments = service.get_patient_appointments(patient_id)

    return [AppointmentResponse.model_validate(a) for a in appointments]


@router.get(
    "/doctor/{doctor_id}",
    summary="Get appointments by doctor",
    description="Returns all appointments for a specific doctor.",
    response_description="Doctor appointments retrieved.",
    response_model=list[AppointmentResponse],
)
def get_doctor_appointments(
    doctor_id: str,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)
    appointments = service.get_doctor_appointments(doctor_id)

    return [AppointmentResponse.model_validate(a) for a in appointments]


# ------------------------------------------------------------------
# Status Transitions
# ------------------------------------------------------------------

@router.put(
    "/{appointment_id}/approve",
    summary="Approve an appointment",
    description="Sets the appointment status to Approved.",
    response_description="Appointment approved.",
    response_model=AppointmentResponse,
)
def approve_appointment(
    appointment_id: str,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)

    try:
        appointment = service.approve_appointment(appointment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return AppointmentResponse.model_validate(appointment)


@router.put(
    "/{appointment_id}/reject",
    summary="Reject an appointment",
    description="Sets the appointment status to Rejected.",
    response_description="Appointment rejected.",
    response_model=AppointmentResponse,
)
def reject_appointment(
    appointment_id: str,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)

    try:
        appointment = service.reject_appointment(appointment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return AppointmentResponse.model_validate(appointment)


@router.put(
    "/{appointment_id}/complete",
    summary="Complete an appointment",
    description="Sets the appointment status to Completed.",
    response_description="Appointment completed.",
    response_model=AppointmentResponse,
)
def complete_appointment(
    appointment_id: str,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)

    try:
        appointment = service.complete_appointment(appointment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return AppointmentResponse.model_validate(appointment)


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------

@router.delete(
    "/{appointment_id}",
    summary="Delete an appointment",
    description="Permanently removes an appointment record.",
    response_description="Appointment deleted.",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_appointment(
    appointment_id: str,
    _current_doctor=Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    service = AppointmentService(db)

    try:
        service.delete_appointment(appointment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
