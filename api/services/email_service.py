import os
from email.message import EmailMessage

# pyrefly: ignore [missing-import]
import aiosmtplib

from api.utils.logger import get_logger

logger = get_logger(__name__)


class EmailService:
    """Sends PDF reports via SMTP using aiosmtplib."""

    def __init__(self) -> None:
        self.smtp_host     = os.getenv("SMTP_HOST")
        self.smtp_port     = int(os.getenv("SMTP_PORT", 587))
        self.smtp_email    = os.getenv("SMTP_EMAIL")
        self.smtp_password = os.getenv("SMTP_PASSWORD")

    async def send_report(
        self,
        recipient_email: str,
        subject: str,
        pdf_bytes: bytes,
        filename: str,
    ) -> None:
        """
        Validate SMTP config, build the email, and send it.
        Raises ValueError if SMTP env vars are not configured.
        """
        if not all([self.smtp_host, self.smtp_email, self.smtp_password]):
            raise ValueError(
                "SMTP configuration is incomplete. "
                "Set SMTP_HOST, SMTP_EMAIL, and SMTP_PASSWORD in your .env file."
            )

        message = EmailMessage()
        message["From"]    = self.smtp_email
        message["To"]      = recipient_email
        message["Subject"] = subject

        message.set_content(
            "Dear Patient,\n\n"
            "Your requested CardioAI Heart Disease Risk Assessment Report has been "
            "generated successfully.\n\n"
            "The attached PDF contains:\n\n"
            "\u2022 Heart disease risk assessment\n"
            "\u2022 Clinical analysis\n"
            "\u2022 AI-generated medical explanation\n"
            "\u2022 Lifestyle recommendations\n"
            "\u2022 Follow-up advice\n\n"
            "Please note that this report is AI-assisted and is intended to support "
            "healthcare professionals. It should not be considered a substitute for "
            "professional medical diagnosis or treatment.\n\n"
            "Thank you for using CardioAI.\n\n"
            "Regards,\n\n"
            "CardioAI Team\n"
            "RV Institute of Technology and Management\n"
            "Bengaluru, Karnataka"
        )

        message.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename=filename,
        )

        logger.info(
            "Sending report email to %s (subject: %s)",
            recipient_email,
            subject,
        )

        await aiosmtplib.send(
            message,
            hostname=self.smtp_host,
            port=self.smtp_port,
            username=self.smtp_email,
            password=self.smtp_password,
            start_tls=True,
        )

        logger.info("Email sent successfully to %s", recipient_email)

    # ------------------------------------------------------------------
    # Appointment Notifications
    # ------------------------------------------------------------------

    async def send_appointment_booked(
        self,
        patient_email: str,
        patient_name: str,
        doctor_name: str,
        preferred_date: str,
        preferred_time: str,
    ) -> None:
        """Notify the patient that their appointment has been booked."""
        subject = "CardioAI – Appointment Booked"
        body = self._build_appointment_body(
            greeting_name=patient_name,
            heading="Appointment Booked Successfully",
            message=(
                f"Your appointment with <strong>Dr. {doctor_name}</strong> has been "
                f"submitted and is currently <strong>Pending</strong> approval."
            ),
            date=preferred_date,
            time=preferred_time,
        )
        await self._send_html(patient_email, subject, body)

    async def send_appointment_approved(
        self,
        patient_email: str,
        patient_name: str,
        doctor_name: str,
        preferred_date: str,
        preferred_time: str,
    ) -> None:
        """Notify the patient that their appointment has been approved."""
        subject = "CardioAI – Appointment Approved"
        body = self._build_appointment_body(
            greeting_name=patient_name,
            heading="Appointment Approved",
            message=(
                f"Your appointment with <strong>Dr. {doctor_name}</strong> "
                f"has been <strong>Approved</strong>. Please be on time."
            ),
            date=preferred_date,
            time=preferred_time,
        )
        await self._send_html(patient_email, subject, body)

    async def send_appointment_rejected(
        self,
        patient_email: str,
        patient_name: str,
        doctor_name: str,
        preferred_date: str,
        preferred_time: str,
    ) -> None:
        """Notify the patient that their appointment has been rejected."""
        subject = "CardioAI – Appointment Rejected"
        body = self._build_appointment_body(
            greeting_name=patient_name,
            heading="Appointment Rejected",
            message=(
                f"Unfortunately, your appointment with <strong>Dr. {doctor_name}</strong> "
                f"has been <strong>Rejected</strong>. Please book a new appointment "
                f"or contact the clinic for further assistance."
            ),
            date=preferred_date,
            time=preferred_time,
        )
        await self._send_html(patient_email, subject, body)

    async def send_appointment_completed(
        self,
        patient_email: str,
        patient_name: str,
        doctor_name: str,
        preferred_date: str,
        preferred_time: str,
    ) -> None:
        """Notify the patient that their appointment has been completed."""
        subject = "CardioAI – Appointment Completed"
        body = self._build_appointment_body(
            greeting_name=patient_name,
            heading="Appointment Completed",
            message=(
                f"Your appointment with <strong>Dr. {doctor_name}</strong> "
                f"has been marked as <strong>Completed</strong>. "
                f"Thank you for choosing CardioAI."
            ),
            date=preferred_date,
            time=preferred_time,
        )
        await self._send_html(patient_email, subject, body)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    async def _send_html(
        self, recipient_email: str, subject: str, html_body: str
    ) -> None:
        """Send a plain HTML email via SMTP.  Shared by all notification methods."""
        if not all([self.smtp_host, self.smtp_email, self.smtp_password]):
            logger.warning(
                "SMTP not configured – skipping email to %s", recipient_email
            )
            return

        message = EmailMessage()
        message["From"]    = self.smtp_email
        message["To"]      = recipient_email
        message["Subject"] = subject
        message.set_content(html_body, subtype="html")

        logger.info(
            "Sending appointment email to %s (subject: %s)",
            recipient_email,
            subject,
        )

        await aiosmtplib.send(
            message,
            hostname=self.smtp_host,
            port=self.smtp_port,
            username=self.smtp_email,
            password=self.smtp_password,
            start_tls=True,
        )

        logger.info("Appointment email sent successfully to %s", recipient_email)

    @staticmethod
    def _build_appointment_body(
        greeting_name: str,
        heading: str,
        message: str,
        date: str,
        time: str,
    ) -> str:
        """Return a simple HTML email body for appointment notifications."""
        return (
            f"<div style='font-family:Arial,sans-serif;max-width:520px;margin:auto'>"
            f"<h2 style='color:#1a73e8'>{heading}</h2>"
            f"<p>Dear {greeting_name},</p>"
            f"<p>{message}</p>"
            f"<table style='border-collapse:collapse;margin:16px 0'>"
            f"<tr><td style='padding:6px 12px;font-weight:bold'>Date</td>"
            f"<td style='padding:6px 12px'>{date}</td></tr>"
            f"<tr><td style='padding:6px 12px;font-weight:bold'>Time</td>"
            f"<td style='padding:6px 12px'>{time}</td></tr>"
            f"</table>"
            f"<p style='color:#555;font-size:13px'>"
            f"This is an automated notification from CardioAI.<br>"
            f"RV Institute of Technology and Management, Bengaluru.</p>"
            f"</div>"
        )