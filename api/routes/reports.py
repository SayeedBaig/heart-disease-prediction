from fastapi import APIRouter, HTTPException

from reports.report_generator import ReportGenerator
from api.services.shared_memory_service import SharedMemoryService

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