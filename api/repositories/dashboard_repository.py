from sqlalchemy.orm import Session, joinedload
from api.models.patient import Patient
from api.models.prediction import Prediction
from api.models.report import Report
from api.models.appointment import Appointment
from api.models.doctor_note import DoctorNote
from api.models.diagnosis import Diagnosis

class DashboardRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_latest_patients(self, limit: int = 20):
        return self.db.query(Patient).order_by(Patient.created_at.desc()).limit(limit).all()

    def get_latest_appointments(self, limit: int = 20):
        return self.db.query(Appointment).options(joinedload(Appointment.patient)).order_by(Appointment.created_at.desc()).limit(limit).all()

    def get_latest_predictions(self, limit: int = 20):
        return self.db.query(Prediction).options(joinedload(Prediction.patient)).order_by(Prediction.created_at.desc()).limit(limit).all()

    def get_latest_reports(self, limit: int = 20):
        return self.db.query(Report).options(joinedload(Report.patient)).order_by(Report.generated_at.desc()).limit(limit).all()

    def get_latest_doctor_notes(self, limit: int = 20):
        return self.db.query(DoctorNote).options(
            joinedload(DoctorNote.diagnosis).joinedload(Diagnosis.patient)
        ).order_by(DoctorNote.created_at.desc()).limit(limit).all()
