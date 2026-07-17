from uuid import UUID
from sqlalchemy.orm import Session
from api.models.doctor_note import DoctorNote

class DoctorNoteRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, data: dict) -> DoctorNote:
        note = DoctorNote(**data)
        self.db.add(note)
        return note
        
    def get_by_diagnosis_id(self, diagnosis_id: UUID) -> list[DoctorNote]:
        return self.db.query(DoctorNote).filter(DoctorNote.diagnosis_id == diagnosis_id).all()
