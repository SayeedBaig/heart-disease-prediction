from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.reports.renderers.pdf_renderer import PdfRenderer
from api.services.email_service import EmailService
from api.services.report_service import ReportService


router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get(
    "/doctor",
    summary="Generate Doctor Report",
    description="Generates a detailed doctor report from the latest prediction.",
    response_description="Doctor report generated successfully.",
)
def get_doctor_report(
    patient_id: str,
    db: Session = Depends(get_db),
):
    return ReportService(db).generate_doctor_report(
        patient_id
    )


@router.get(
    "/patient",
    summary="Generate Patient Report",
    description="Generates a simplified patient-facing report from the latest prediction.",
    response_description="Patient report generated successfully.",
)
def get_patient_report(
    patient_id: str,
    db: Session = Depends(get_db),
):
    return ReportService(db).generate_patient_report(
        patient_id
    )

@router.get(
    "/doctor/pdf",
    summary="Generate Doctor PDF",
    description="Generates and downloads a detailed doctor report PDF for the given patient.",
    response_description="Doctor PDF generated successfully.",
)
def download_doctor_report(
    patient_id: str,
    db: Session = Depends(get_db),
):
    report = ReportService(db).generate_doctor_report(patient_id)

    pdf = PdfRenderer().render(report)

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
            f'attachment; filename="doctor_report_{patient_id}.pdf"'
        },
    )

@router.get(
    "/patient/pdf",
    summary="Generate Patient PDF",
    description="Generates and downloads a patient-facing report PDF for the given patient.",
    response_description="Patient PDF generated successfully.",
)
def download_patient_report(
    patient_id: str,
    db: Session = Depends(get_db),
):
    report = ReportService(db).generate_patient_report(patient_id)

    pdf = PdfRenderer().render(report)

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
            f'attachment; filename="patient_report_{patient_id}.pdf"'
        },
    )
@router.post(
    "/patient/email",
    summary="Email Patient Report",
    description="Generates a patient PDF report and emails it to the registered patient.",
    response_description="Email sent successfully.",
)
async def email_patient_report(
    patient_id: str,
    db: Session = Depends(get_db),
):

    report = ReportService(db).generate_patient_report(
        patient_id
    )

    pdf_buffer = PdfRenderer().render(report)

    patient = report.get("patient")

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient information not found.",
        )

    await EmailService().send_report(
        recipient_email=patient["email"],
        subject="CardioAI Medical Report – Heart Disease Risk Assessment",
        pdf_bytes=pdf_buffer.getvalue(),
        filename=f"patient_report_{patient_id}.pdf",
    )

    return {
        "success": True,
        "message": "Patient report emailed successfully.",
    }


@router.post(
    "/doctor/email",
    summary="Email Doctor Report",
    description="Generates a doctor PDF report and emails it to the registered patient.",
    response_description="Email sent successfully.",
)
async def email_doctor_report(
    patient_id: str,
    db: Session = Depends(get_db),
):

    report = ReportService(db).generate_doctor_report(
        patient_id
    )

    pdf_buffer = PdfRenderer().render(report)

    patient = report.get("patient")

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient information not found.",
        )

    await EmailService().send_report(
        recipient_email=patient["email"],
        subject="CardioAI Doctor Report – Heart Disease Risk Assessment",
        pdf_bytes=pdf_buffer.getvalue(),
        filename=f"doctor_report_{patient_id}.pdf",
    )

    return {
        "success": True,
        "message": "Doctor report emailed successfully.",
    }