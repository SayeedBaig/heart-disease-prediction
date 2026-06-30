from typing import Dict, Any

from reports.doctor_report import DoctorReport
from reports.patient_report import PatientReport


class ReportGenerator:

    def __init__(self):

        self.doctor_report = DoctorReport()

        self.patient_report = PatientReport()

    def generate_doctor_report(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        digital_twin: Dict[str, Any],
    ):

        return self.doctor_report.generate(
            prediction,
            explanation,
            digital_twin,
        )

    def generate_patient_report(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
    ):

        return self.patient_report.generate(
            prediction,
            explanation,
        )