from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.models.appointment import Appointment, AppointmentStatus
from api.repositories.doctor_repository import DoctorRepository
from api.repositories.patient_repository import PatientRepository
from api.repositories.prediction_repository import PredictionRepository
from api.utils.auth import verify_access_token


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/patients/login")


def get_current_actor(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """Return the authenticated patient or doctor without changing JWT contracts."""
    payload = verify_access_token(token)
    role = payload.get("role") if payload else None
    subject = payload.get("sub") if payload else None

    if not subject or role not in {"patient", "doctor"}:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")

    if role == "patient":
        try:
            actor = PatientRepository(db).get_by_id(int(subject))
        except (TypeError, ValueError):
            actor = None
    else:
        actor = DoctorRepository(db).get_doctor_by_id(subject)

    if actor is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated account was not found.")

    return role, actor


def resolve_patient(patient_identifier: str, db: Session):
    """Resolve either the public patient ID or the internal numeric ID."""
    repository = PatientRepository(db)
    patient = repository.get_by_public_id(patient_identifier)
    if patient is None:
        try:
            patient = repository.get_by_id(int(patient_identifier))
        except (TypeError, ValueError):
            patient = None
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")
    return patient


def ensure_patient_access(patient, actor, db: Session) -> None:
    """Allow patients to view only themselves; authenticated doctors may view clinical records."""
    role, account = actor
    if role == "patient":
        if account.id != patient.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only access your own records.")
        return

    # The clinician portal is an organisation-wide workspace: a verified doctor
    # can view every registered patient's clinical record. Patient access remains
    # self-only and is enforced above for every patient-facing route.
    return


def ensure_prediction_access(prediction_id: int, actor, db: Session):
    prediction = PredictionRepository(db).get_by_id(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    patient = PatientRepository(db).get_by_id(prediction.patient_id)
    ensure_patient_access(patient, actor, db)
    return prediction
