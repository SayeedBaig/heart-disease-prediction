from typing import Any, Dict, Optional

from api.reports.report_builder import ReportBuilder
from api.reports.renderers.json_renderer import JsonRenderer


class ReportGenerator:
    """
    Public facade consumed by api/routes/reports.py.

    Public method signatures are unchanged — existing routes
    require no modification.

    Internally delegates data assembly to ReportBuilder and
    format rendering to JsonRenderer.
    """

    def __init__(self) -> None:
        self._builder = ReportBuilder()
        self._renderer = JsonRenderer()

    def generate_doctor_report(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        digital_twin: Dict[str, Any],
        patient: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        report_data = self._builder.build_doctor_report(
            prediction,
            explanation,
            digital_twin,
            patient,
        )

        return self._renderer.render(report_data)

    def generate_patient_report(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        patient: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        report_data = self._builder.build_patient_report(
            prediction,
            explanation,
            patient,
        )

        return self._renderer.render(report_data)