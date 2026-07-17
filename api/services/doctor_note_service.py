from uuid import UUID
from sqlalchemy.orm import Session
from api.repositories.doctor_note_repository import DoctorNoteRepository
from api.repositories.diagnosis_repository import DiagnosisRepository

class DoctorNoteService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = DoctorNoteRepository(db)
        self.diagnosis_repo = DiagnosisRepository(db)

    def add_note(self, data: dict, doctor_id: UUID):
        diagnosis = self.diagnosis_repo.get_by_id(data["diagnosis_id"])
        if not diagnosis:
            raise ValueError("Diagnosis not found.")
        if str(diagnosis.doctor_id) != str(doctor_id):
            raise ValueError("Unauthorized to add notes to this diagnosis.")
            
        try:
            note = self.repo.create(data)
            self.db.commit()
            self.db.refresh(note)
        except Exception:
            self.db.rollback()
            raise
        return note
        
    def get_notes(self, diagnosis_id: UUID, doctor_id: UUID):
        diagnosis = self.diagnosis_repo.get_by_id(diagnosis_id)
        if not diagnosis:
            raise ValueError("Diagnosis not found.")
        if str(diagnosis.doctor_id) != str(doctor_id):
            raise ValueError("Unauthorized.")
        return self.repo.get_by_diagnosis_id(diagnosis_id)
