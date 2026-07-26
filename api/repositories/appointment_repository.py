from sqlalchemy.orm import Session

from api.models.appointment import Appointment, AppointmentStatus


class AppointmentRepository:

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_appointment(self, appointment_data: dict) -> Appointment:
        """Add a new appointment record to the session."""
        appointment = Appointment(**appointment_data)

        self.db.add(appointment)

        return appointment

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, appointment_id: str) -> Appointment | None:
        """Return an appointment by UUID primary key, or None if not found."""
        return (
            self.db.query(Appointment)
            .filter(Appointment.appointment_id == appointment_id)
            .first()
        )

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Appointment]:
        """Return a paginated list of all appointments, newest first."""
        return (
            self.db.query(Appointment)
            .order_by(Appointment.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_patient(self, patient_id: int) -> list[Appointment]:
        """Return all appointments for a given patient."""
        return (
            self.db.query(Appointment)
            .filter(Appointment.patient_id == patient_id)
            .order_by(Appointment.created_at.desc())
            .all()
        )

    def get_by_doctor(self, doctor_id: str) -> list[Appointment]:
        """Return all appointments for a given doctor."""
        return (
            self.db.query(Appointment)
            .filter(Appointment.doctor_id == doctor_id)
            .order_by(Appointment.created_at.desc())
            .all()
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_status(
        self, appointment: Appointment, status: AppointmentStatus
    ) -> Appointment:
        """Update the status of an existing appointment."""
        appointment.status = status

        return appointment

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, appointment: Appointment) -> None:
        """Mark an appointment record for deletion in the session."""
        self.db.delete(appointment)
