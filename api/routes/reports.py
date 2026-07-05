from fastapi import APIRouter, HTTPException

from reports.report_generator import ReportGenerator
from api.services.shared_memory_service import SharedMemoryService

from fastapi.responses import StreamingResponse

from api.reports.renderers.pdf_renderer import PdfRenderer
from api.services.report_service import ReportService
from api.database.session import get_db
from sqlalchemy.orm import Session
from fastapi import Depends

router = APIRouter(prefix="/reports", tags=["Reports"])

report_generator = ReportGenerator()
shared_memory = SharedMemoryService()


@router.get("/doctor")
def get_doctor_report():
    """
    Generate the latest doctor report.
    """

    prediction = shared_memory.get("latest_prediction")
    explanation = shared_memory.get("latest_explanation")
    digital_twin = shared_memory.get("latest_digital_twin")

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="No prediction available. Run /predict first."
        )

    return report_generator.generate_doctor_report(
        prediction,
        explanation,
        digital_twin,
    )


@router.get("/patient")
def get_patient_report():
    """
    Generate the latest patient report.
    """

    prediction = shared_memory.get("latest_prediction")
    explanation = shared_memory.get("latest_explanation")
    digital_twin = shared_memory.get("latest_digital_twin")

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="No prediction available. Run /predict first."
        )

    return report_generator.generate_patient_report(
        prediction,
        explanation,
    )

@router.get("/doctor/pdf")
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

@router.get("/patient/pdf")
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