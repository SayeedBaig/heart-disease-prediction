from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.database.session import get_db
from api.utils.auth import get_current_doctor
from api.schemas.doctor_note import DoctorNoteCreate, DoctorNoteResponse
from api.services.doctor_note_service import DoctorNoteService

router = APIRouter(prefix="/notes", tags=["Doctor Notes"])

@router.post("/", summary="Add a doctor note", response_model=DoctorNoteResponse)
def add_note(
    data: DoctorNoteCreate,
    db: Session = Depends(get_db),
    current_doctor=Depends(get_current_doctor)
):
    try:
        return DoctorNoteService(db).add_note(data.model_dump(), current_doctor.doctor_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/diagnosis/{diagnosis_id}", summary="Get notes for diagnosis", response_model=list[DoctorNoteResponse])
def get_notes(
    diagnosis_id: UUID,
    db: Session = Depends(get_db),
    current_doctor=Depends(get_current_doctor)
):
    try:
        return DoctorNoteService(db).get_notes(diagnosis_id, current_doctor.doctor_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
