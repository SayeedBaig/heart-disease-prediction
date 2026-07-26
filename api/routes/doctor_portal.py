"""Read-only doctor portal APIs for authenticated clinical staff."""
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.models.appointment import Appointment, AppointmentStatus
from api.models.patient import Patient
from api.models.prediction import Prediction
from api.reports.renderers.pdf_renderer import PdfRenderer
from api.repositories.diagnosis_repository import DiagnosisRepository
from api.services.report_service import ReportService
from api.utils.auth import get_current_doctor
from api.utils.authorization import ensure_prediction_access, ensure_patient_access, get_current_actor


router = APIRouter(prefix="/doctor", tags=["Doctor Portal"])
_ASSIGNED_STATUSES = (AppointmentStatus.PENDING, AppointmentStatus.APPROVED, AppointmentStatus.COMPLETED)


def _age(value):
    if not value:
        return None
    today = date.today()
    return today.year - value.year - ((today.month, today.day) < (value.month, value.day))


def _latest_prediction(db: Session, patient_id: int):
    return db.query(Prediction).filter(Prediction.patient_id == patient_id).order_by(Prediction.created_at.desc()).first()


def _prediction_payload(prediction):
    if prediction is None:
        return None
    return {
        "prediction_id": prediction.id,
        "risk_score": prediction.risk_percentage,
        "risk_level": prediction.risk_level,
        "confidence_score": prediction.confidence,
        "prediction_date": prediction.created_at,
        "clinical_level": prediction.clinical_level,
        "ecg_level": prediction.ecg_level,
        "echo_level": prediction.echo_level,
    }


def _visible_patients(db: Session):
    """Return every registered patient for the organisation-wide doctor workspace."""
    return db.query(Patient).order_by(Patient.created_at.desc()).all()


@router.get("/dashboard", summary="Doctor dashboard", description="Dashboard statistics for all registered patients.")
def doctor_dashboard(db: Session = Depends(get_db), doctor=Depends(get_current_doctor)):
    patients = _visible_patients(db)
    latest = [_latest_prediction(db, patient.id) for patient in patients]
    levels = {"low": 0, "medium": 0, "high": 0}
    for prediction in latest:
        if prediction:
            key = (prediction.risk_level or "").lower()
            if key in levels:
                levels[key] += 1
    return {
        "total_assigned_patients": len(patients),
        "total_reports": sum(1 for prediction in latest if prediction),
        "low_risk_patients": levels["low"],
        "medium_risk_patients": levels["medium"],
        "high_risk_patients": levels["high"],
        "latest_predictions": [_prediction_payload(prediction) for prediction in latest if prediction][:10],
    }


@router.get("/patients", summary="List patients", description="Search, filter, sort, and paginate all registered patients for the authenticated doctor.")
def doctor_patients(
    search: str | None = None,
    risk_level: str | None = None,
    sort: str = Query("latest_prediction", pattern="^(name|latest_prediction|risk)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    doctor=Depends(get_current_doctor),
):
    rows = []
    needle = (search or "").strip().lower()
    for patient in _visible_patients(db):
        prediction = _latest_prediction(db, patient.id)
        if needle and needle not in patient.full_name.lower() and needle not in patient.email.lower():
            continue
        if risk_level and (not prediction or (prediction.risk_level or "").lower() != risk_level.lower()):
            continue
        rows.append({
            "patient_id": patient.id, "public_patient_id": patient.patient_id, "name": patient.full_name,
            "age": _age(patient.date_of_birth), "gender": patient.gender, "email": patient.email,
            "latest_prediction": _prediction_payload(prediction),
        })
    if sort == "name":
        rows.sort(key=lambda row: row["name"].lower())
    elif sort == "risk":
        rows.sort(key=lambda row: row["latest_prediction"]["risk_score"] if row["latest_prediction"] else -1, reverse=True)
    else:
        rows.sort(key=lambda row: row["latest_prediction"]["prediction_date"] if row["latest_prediction"] else datetime.min, reverse=True)
    start = (page - 1) * page_size
    return {"items": rows[start:start + page_size], "page": page, "page_size": page_size, "total": len(rows)}


@router.get("/patients/{patient_id}", summary="Patient details", description="Complete read-only patient record for the authenticated doctor.")
def doctor_patient_detail(patient_id: int, db: Session = Depends(get_db), actor=Depends(get_current_actor)):
    if actor[0] != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access is required.")
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    ensure_patient_access(patient, actor, db)
    predictions = db.query(Prediction).filter(Prediction.patient_id == patient.id).order_by(Prediction.created_at.desc()).all()

    # Populate clinical_information from the most recent completed diagnosis
    diagnosis_repo = DiagnosisRepository(db)
    latest_diagnosis = diagnosis_repo.get_latest_by_patient_id(patient.id)
    clinical_information = None
    ecg_info = None
    echo_info = None
    if latest_diagnosis:
        clinical_information = latest_diagnosis.clinical_data or {}
        if latest_diagnosis.ecg_path:
            ecg_info = {"path": latest_diagnosis.ecg_path, "level": predictions[0].ecg_level if predictions else None}
        if latest_diagnosis.echo_path:
            echo_info = {"path": latest_diagnosis.echo_path, "level": predictions[0].echo_level if predictions else None}
    elif predictions:
        # Fallback: show ecg/echo level from prediction even without diagnosis record
        ecg_info = {"level": predictions[0].ecg_level} if predictions[0].ecg_level else None
        echo_info = {"level": predictions[0].echo_level} if predictions[0].echo_level else None

    return {
        "patient": {
            "patient_id": patient.id,
            "public_patient_id": patient.patient_id,
            "name": patient.full_name,
            "age": _age(patient.date_of_birth),
            "gender": patient.gender,
            "email": patient.email,
            "phone_number": patient.phone,
        },
        "clinical_information": clinical_information,
        "latest_prediction": _prediction_payload(predictions[0]) if predictions else None,
        "prediction_history": [_prediction_payload(p) for p in predictions],
        "reports": [
            {
                "report_id": p.id,
                "generated_date": p.created_at,
                "risk_level": p.risk_level,
                "view_url": f"/doctor/reports/{p.id}",
                "download_url": f"/doctor/reports/{p.id}/download",
            }
            for p in predictions
        ],
        "digital_twin": None,
        "ecg": ecg_info,
        "echo": echo_info,
    }


@router.get("/reports", summary="List patient reports", description="Paginated prediction-backed reports for all registered patients.")
def doctor_reports(
    search: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    doctor=Depends(get_current_doctor),
):
    needle = (search or "").strip().lower()
    items = []
    for patient in _visible_patients(db):
        if needle and needle not in patient.full_name.lower() and needle not in patient.email.lower():
            continue
        for prediction in db.query(Prediction).filter(Prediction.patient_id == patient.id).order_by(Prediction.created_at.desc()).all():
            items.append({"report_id": prediction.id, "patient_id": patient.id, "patient_name": patient.full_name, "generated_date": prediction.created_at, "risk_level": prediction.risk_level, "view_url": f"/doctor/reports/{prediction.id}", "download_url": f"/doctor/reports/{prediction.id}/download"})
    items.sort(key=lambda item: item["generated_date"], reverse=True)
    start = (page - 1) * page_size
    return {"items": items[start:start + page_size], "page": page, "page_size": page_size, "total": len(items)}


@router.get("/reports/{report_id}", summary="View doctor report", description="Detailed generated report for an assigned patient's prediction.")
def doctor_report(report_id: int, db: Session = Depends(get_db), actor=Depends(get_current_actor)):
    if actor[0] != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access is required.")
    ensure_prediction_access(report_id, actor, db)
    return ReportService(db).generate_doctor_report(report_id)


@router.get("/reports/{report_id}/download", summary="Download doctor report", description="PDF report for an assigned patient's prediction.")
def download_doctor_report(report_id: int, db: Session = Depends(get_db), actor=Depends(get_current_actor)):
    if actor[0] != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access is required.")
    ensure_prediction_access(report_id, actor, db)
    pdf = PdfRenderer().render(ReportService(db).generate_doctor_report(report_id))
    return StreamingResponse(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="doctor_report_{report_id}.pdf"'})
