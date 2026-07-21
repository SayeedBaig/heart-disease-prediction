from sqlalchemy.orm import Session

from api.repositories.patient_repository import PatientRepository
from api.utils.auth import hash_password


class PatientRegistrationService:
    def __init__(self, db: Session):
        self.db = db
        self.patient_repository = PatientRepository(db)

    def register_patient(self, patient_data: dict):
        existing_patient = self.patient_repository.get_by_email(
            patient_data["email"]
        )

        if existing_patient:
            raise ValueError("Patient with this email already exists.")

        last_patient = self.patient_repository.get_last_patient()

        if last_patient:
            last_number = int(last_patient.patient_id.replace("PT", ""))
            patient_number = last_number + 1
        else:
            patient_number = 1

        patient_data["patient_id"] = f"PT{patient_number:06d}"
        patient_data["password_hash"] = hash_password(
        patient_data.pop("password")
    )

        try:
            patient = self.patient_repository.create(patient_data)
            self.db.commit()
            self.db.refresh(patient)
        except Exception:
            self.db.rollback()
            raise

        return patient