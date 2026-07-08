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