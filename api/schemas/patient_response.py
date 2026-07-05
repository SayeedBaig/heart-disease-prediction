from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class PatientRegisterResponse(BaseModel):
    """
    Response returned after a successful patient registration.
    Reads directly from the SQLAlchemy Patient ORM object via from_attributes.
    The 'message' field is injected by the route layer since it is not on the ORM model.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: str
    full_name: str
    email: EmailStr
    message: str
    created_at: datetime