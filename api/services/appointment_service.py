import asyncio

from sqlalchemy.orm import Session

from api.models.appointment import Appointment, AppointmentStatus
from api.repositories.appointment_repository import AppointmentRepository
from api.services.email_service import EmailService
from api.utils.logger import get_logger

logger = get_logger(__name__)


class AppointmentService:
    def __init__(self, db: Session):
        self.db = db
        self.appointment_repo = AppointmentRepository(db)
        self.email_service = EmailService()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_appointment(self, appointment_data: dict):
        """Book a new appointment.  Returns the persisted record."""
        try:
            appointment = self.appointment_repo.create_appointment(
                appointment_data
            )
            self.db.commit()
            self.db.refresh(appointment)
        except Exception:
            self.db.rollback()
            raise

        self._notify(appointment, "booked")

        return appointment

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_appointment(self, appointment_id: str):
        """Return a single appointment or raise if not found."""
        appointment = self.appointment_repo.get_by_id(appointment_id)

        if not appointment:
            raise ValueError("Appointment not found.")

        return appointment

    def get_all_appointments(self, skip: int = 0, limit: int = 100):
        """Return a paginated list of all appointments."""
        return self.appointment_repo.get_all(skip=skip, limit=limit)

    def get_patient_appointments(self, patient_id: int):
        """Return all appointments for a specific patient."""
        return self.appointment_repo.get_by_patient(patient_id)

    def get_doctor_appointments(self, doctor_id: str):
        """Return all appointments for a specific doctor."""
        return self.appointment_repo.get_by_doctor(doctor_id)

    # ------------------------------------------------------------------
    # Status Transitions
    # ------------------------------------------------------------------

    def approve_appointment(self, appointment_id: str):
        """Set appointment status to Approved.  Raises ValueError if not found."""
        return self._transition_status(
            appointment_id, AppointmentStatus.APPROVED
        )

    def reject_appointment(self, appointment_id: str):
        """Set appointment status to Rejected.  Raises ValueError if not found."""
        return self._transition_status(
            appointment_id, AppointmentStatus.REJECTED
        )

    def complete_appointment(self, appointment_id: str):
        """Set appointment status to Completed.  Raises ValueError if not found."""
        return self._transition_status(
            appointment_id, AppointmentStatus.COMPLETED
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_appointment(self, appointment_id: str):
        """Delete an appointment by UUID.  Raises ValueError if not found."""
        appointment = self.appointment_repo.get_by_id(appointment_id)

        if not appointment:
            raise ValueError("Appointment not found.")

        try:
            self.appointment_repo.delete(appointment)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transition_status(self, appointment_id: str, status: AppointmentStatus):
        """Shared helper for all status transitions."""
        appointment = self.appointment_repo.get_by_id(appointment_id)

        if not appointment:
            raise ValueError("Appointment not found.")

        try:
            appointment = self.appointment_repo.update_status(
                appointment, status
            )
            self.db.commit()
            self.db.refresh(appointment)
        except Exception:
            self.db.rollback()
            raise

        event_map = {
            AppointmentStatus.APPROVED: "approved",
            AppointmentStatus.REJECTED: "rejected",
            AppointmentStatus.COMPLETED: "completed",
        }
        self._notify(appointment, event_map[status])

        return appointment

    def _notify(self, appointment: Appointment, event: str) -> None:
        """Fire-and-forget email notification for an appointment event."""
        try:
            patient = appointment.patient
            doctor = appointment.doctor

            if not patient or not doctor:
                logger.warning("Cannot send email – missing patient/doctor relationship.")
                return

            method_map = {
                "booked": self.email_service.send_appointment_booked,
                "approved": self.email_service.send_appointment_approved,
                "rejected": self.email_service.send_appointment_rejected,
                "completed": self.email_service.send_appointment_completed,
            }

            coro = method_map[event](
                patient_email=patient.email,
                patient_name=patient.full_name,
                doctor_name=doctor.full_name,
                preferred_date=str(appointment.preferred_date),
                preferred_time=str(appointment.preferred_time),
            )

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)

        except Exception as exc:
            logger.error("Failed to send %s email: %s", event, exc)
