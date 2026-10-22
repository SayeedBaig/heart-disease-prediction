from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.reports.renderers.pdf_renderer import PdfRenderer
from api.services.email_service import EmailService
from api.services.report_service import ReportService
from api.utils.authorization import ensure_prediction_access, get_current_actor, get_current_chat_session
from api.utils.auth import get_current_patient
from api.services.chat_session_service import replace_report


router = APIRouter(prefix="/reports", tags=["Reports"])


def _index_report(db: Session, session, report: dict) -> None:
    try:
        replace_report(db, session, report)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Report context could not be prepared for the assistant.") from exc


@router.post(
    "/digital-twin/email",
    summary="Email Digital Twin Report",
    description="Generates the authenticated patient's Digital Twin report using the shared CardioAI PDF template and emails it to their registered address.",
)
async def email_digital_twin_report(
    simulation: dict = Body(...),
    current_patient=Depends(get_current_patient),
    session=Depends(get_current_chat_session),
    db: Session = Depends(get_db),
):
    risk = simulation.get("risk", {})
    # clinical_alerts may be present in the simulation payload (from the frontend
    # buildReport function).  We pass them through so the PDF includes them.
    # Both baseline and per-scenario alerts may be present.
    clinical_alerts = simulation.get("clinical_alerts") or []
    report = {
        "report_type": "patient",
        "patient": {
            "patient_id": current_patient.patient_id,
            "full_name": current_patient.full_name,
            "email": current_patient.email,
            "phone": current_patient.phone,
            "gender": current_patient.gender,
            "date_of_birth": str(current_patient.date_of_birth),
        },
        "risk_level": risk.get("level"),
        "risk_percentage": risk.get("score"),
        "summary": simulation.get("summary"),
        "details": "Digital Twin simulation generated from the patient's current clinical baseline.",
        "digital_twin": {"scenarios": simulation.get("scenarios", []), "parameters": simulation.get("metrics", [])},
        "lifestyle_recommendations": simulation.get("recommendations", []),
        "follow_up_advice": ["Review this simulation with your treating clinician before making health decisions."],
        "clinical_alerts": clinical_alerts,
    }
    _index_report(db, session, report)
    pdf_buffer = PdfRenderer().render(report)
    await EmailService().send_report(
        recipient_email=current_patient.email,
        subject="CardioAI Digital Twin Report",
        pdf_bytes=pdf_buffer.getvalue(),
        filename=f"digital_twin_report_{current_patient.patient_id}.pdf",
    )
    return {"success": True, "message": "Digital Twin Report sent successfully."}



@router.get(
    "/{prediction_id}/doctor",
    summary="Generate Doctor Report",
    description="Generates a detailed doctor report from a prediction.",
    response_description="Doctor report generated successfully.",
)
def get_doctor_report(
    prediction_id: int,
    db: Session = Depends(get_db),
    actor=Depends(get_current_actor),
    session=Depends(get_current_chat_session),
):
    if actor[0] != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access is required for this report.")
    ensure_prediction_access(prediction_id, actor, db)
    report = ReportService(db).generate_doctor_report(prediction_id)
    _index_report(db, session, report)
    return report


@router.get(
    "/{prediction_id}/patient",
    summary="Generate Patient Report",
    description="Generates a simplified patient-facing report from a prediction.",
    response_description="Patient report generated successfully.",
)
def get_patient_report(
    prediction_id: int,
    db: Session = Depends(get_db),
    actor=Depends(get_current_actor),
    session=Depends(get_current_chat_session),
):
    ensure_prediction_access(prediction_id, actor, db)
    report = ReportService(db).generate_patient_report(prediction_id)
    _index_report(db, session, report)
    return report


@router.get(
    "/{prediction_id}/doctor/pdf",
    summary="Generate Doctor PDF",
    description="Generates and downloads a detailed doctor report PDF for the given prediction.",
    response_description="Doctor PDF generated successfully.",
)
def download_doctor_report(
    prediction_id: int,
    db: Session = Depends(get_db),
    actor=Depends(get_current_actor),
    session=Depends(get_current_chat_session),
):
    if actor[0] != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access is required for this report.")
    ensure_prediction_access(prediction_id, actor, db)
    report = ReportService(db).generate_doctor_report(prediction_id)
    _index_report(db, session, report)
    pdf = PdfRenderer().render(report)

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="doctor_report_{prediction_id}.pdf"'
        },
    )


@router.get(
    "/{prediction_id}/patient/pdf",
    summary="Generate Patient PDF",
    description="Generates and downloads a patient-facing report PDF for the given prediction.",
    response_description="Patient PDF generated successfully.",
)
def download_patient_report(
    prediction_id: int,
    db: Session = Depends(get_db),
    actor=Depends(get_current_actor),
    session=Depends(get_current_chat_session),
):
    ensure_prediction_access(prediction_id, actor, db)
    report = ReportService(db).generate_patient_report(prediction_id)
    _index_report(db, session, report)
    pdf = PdfRenderer().render(report)

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="patient_report_{prediction_id}.pdf"'
        },
    )


@router.post(
    "/{prediction_id}/patient/email",
    summary="Email Patient Report",
    description="Generates a patient PDF report and emails it to the registered patient.",
    response_description="Email sent successfully.",
)
async def email_patient_report(
    prediction_id: int,
    db: Session = Depends(get_db),
    actor=Depends(get_current_actor),
    session=Depends(get_current_chat_session),
):
    if actor[0] != "patient":
        raise HTTPException(status_code=403, detail="Patients may email only their own reports.")
    ensure_prediction_access(prediction_id, actor, db)
    report = ReportService(db).generate_patient_report(prediction_id)
    _index_report(db, session, report)
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
        filename=f"patient_report_{prediction_id}.pdf",
    )

    return {
        "success": True,
        "message": "Patient report emailed successfully.",
    }


@router.post(
    "/{prediction_id}/doctor/email",
    summary="Email Doctor Report",
    description="Generates a doctor PDF report and emails it to the registered patient.",
    response_description="Email sent successfully.",
)
async def email_doctor_report(
    prediction_id: int,
    db: Session = Depends(get_db),
    actor=Depends(get_current_actor),
    session=Depends(get_current_chat_session),
):
    if actor[0] != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access is required for this report.")
    ensure_prediction_access(prediction_id, actor, db)
    report = ReportService(db).generate_doctor_report(prediction_id)
    _index_report(db, session, report)
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
        filename=f"doctor_report_{prediction_id}.pdf",
    )

    return {
        "success": True,
        "message": "Doctor report emailed successfully.",
    }
