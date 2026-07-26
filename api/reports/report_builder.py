from typing import Any, Dict, Optional

from reports.doctor_report import DoctorReport
from reports.patient_report import PatientReport


class ReportBuilder:
    """
    Assembles raw report data by delegating to the existing
    DoctorReport and PatientReport classes.

    Owns no report logic — all field construction stays in the
    original report classes to avoid duplication.
    """

    def __init__(self) -> None:
        self._doctor = DoctorReport()
        self._patient = PatientReport()

    def build_doctor_report(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        digital_twin: Dict[str, Any],
        patient: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._doctor.generate(
            prediction,
            explanation,
            digital_twin,
            patient,
        )

    def build_patient_report(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        patient: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._patient.generate(
            prediction,
            explanation,
            patient,
        )
