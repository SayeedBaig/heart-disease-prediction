from api.repositories.patient_repository import PatientRepository


class PatientRegistrationService:
    def __init__(self, db):
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

        return self.patient_repository.create(patient_data)