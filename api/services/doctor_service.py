from sqlalchemy.orm import Session

from api.repositories.doctor_repository import DoctorRepository
from api.utils.security import hash_password, verify_password
from api.utils.auth import create_access_token


class DoctorService:
    def __init__(self, db: Session):
        self.db = db
        self.doctor_repository = DoctorRepository(db)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_doctor(self, doctor_data: dict):
        """Register a new doctor.  Raises ValueError on duplicate email."""
        existing = self.doctor_repository.get_doctor_by_email(
            doctor_data["email"]
        )

        if existing:
            raise ValueError("A doctor with this email already exists.")

        doctor_data["password_hash"] = hash_password(doctor_data.pop("password"))

        try:
            doctor = self.doctor_repository.create_doctor(doctor_data)
            self.db.commit()
            self.db.refresh(doctor)
        except Exception:
            self.db.rollback()
            raise

        return doctor

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    def login_doctor(self, email: str, password: str) -> dict:
        """Authenticate a doctor and return a JWT + profile.

        Raises ValueError on bad credentials.
        """
        doctor = self.doctor_repository.get_doctor_by_email(email)

        if not doctor or not verify_password(password, doctor.password_hash):
            raise ValueError("Invalid email or password.")

        token = create_access_token(
            data={"sub": str(doctor.doctor_id), "email": doctor.email}
        )

        return {"access_token": token, "doctor": doctor}

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_doctor_profile(self, doctor_id: str):
        """Return a doctor by UUID or None."""
        return self.doctor_repository.get_doctor_by_id(doctor_id)

    def update_doctor_profile(self, doctor_id: str, updates: dict):
        """Update and return the doctor profile.

        Raises ValueError if the doctor is not found.
        """
        doctor = self.doctor_repository.get_doctor_by_id(doctor_id)

        if not doctor:
            raise ValueError("Doctor not found.")

        # Strip None values so only supplied fields are updated
        clean_updates = {k: v for k, v in updates.items() if v is not None}

        if not clean_updates:
            return doctor

        try:
            doctor = self.doctor_repository.update_doctor(doctor, clean_updates)
            self.db.commit()
            self.db.refresh(doctor)
        except Exception:
            self.db.rollback()
            raise

        return doctor

    # ------------------------------------------------------------------
    # Admin / Listing
    # ------------------------------------------------------------------

    def list_doctors(self, skip: int = 0, limit: int = 100):
        """Return a paginated list of doctors."""
        return self.doctor_repository.list_doctors(skip=skip, limit=limit)

    def delete_doctor(self, doctor_id: str):
        """Delete a doctor by UUID.  Raises ValueError if not found."""
        doctor = self.doctor_repository.get_doctor_by_id(doctor_id)

        if not doctor:
            raise ValueError("Doctor not found.")

        try:
            self.doctor_repository.delete_doctor(doctor)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
