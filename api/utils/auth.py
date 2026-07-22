import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.repositories.doctor_repository import DoctorRepository
from api.repositories.patient_repository import PatientRepository

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "change_this_to_a_long_random_secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/doctors/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ------------------------------------------------------------------
# JWT Utilities
# ------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT with an expiry claim."""
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_access_token(token: str) -> dict | None:
    """Decode and validate a JWT.  Returns the payload or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
def hash_password(password: str) -> str:
    """Hash a plain-text password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

# ------------------------------------------------------------------
# FastAPI Auth Dependency
# ------------------------------------------------------------------

def get_current_doctor(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """Validate the JWT and return the authenticated doctor.

    Used as a FastAPI dependency on protected endpoints.
    """
    payload = verify_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    doctor_id = payload.get("sub")

    if payload.get("role") != "doctor" or doctor_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    try:
        doctor_id = UUID(doctor_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid doctor token.",
        )

    doctor_repo = DoctorRepository(db)
    doctor = doctor_repo.get_doctor_by_id(doctor_id)

    if doctor is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Doctor not found.",
        )

    return doctor


def get_current_patient(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """Validate a patient JWT and return the authenticated patient."""
    payload = verify_access_token(token)

    if payload is None or payload.get("role") != "patient":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired patient token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        patient_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid patient token.",
        )

    patient = PatientRepository(db).get_by_id(patient_id)

    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found.",
        )

    return patient
