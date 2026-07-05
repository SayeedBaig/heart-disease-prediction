from sqlalchemy.orm import Session

from api.models.patient import Patient


class PatientRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str):
        return (
            self.db.query(Patient)
            .filter(Patient.email == email)
            .first()
        )

    def create(self, patient_data: dict):
        patient = Patient(**patient_data)

        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)

        return patient

    def get_or_create(self, patient_data: dict):
        patient = self.get_by_email(patient_data["email"])

        if patient:
            return patient

        return self.create(patient_data)

    def get_last_patient(self):
        return (
            self.db.query(Patient)
            .order_by(Patient.id.desc())
            .first()
        )  

    def get_by_public_id(self, patient_id: str):
        return (
            self.db.query(Patient)
            .filter(Patient.patient_id == patient_id)
            .first()
        )