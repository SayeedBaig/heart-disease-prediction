from io import BytesIO
from typing import Any, Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate

from api.reports.pdf_formatter import PdfFormatter, build_styles

_MARGIN = 20 * mm


class PdfRenderer:

    def render(self, report_data: Dict[str, Any]) -> BytesIO:
        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=_MARGIN,
            rightMargin=_MARGIN,
            topMargin=_MARGIN,
            bottomMargin=_MARGIN,
        )

        styles      = build_styles()
        fmt         = PdfFormatter(styles)
        report_type = (report_data.get("report_type") or "report").lower()

        elements = []

        # ── Header ──────────────────────────────────────────────────────
        elements.extend(fmt.format_header(report_type.title()))
        elements.extend(fmt.section(
            "Academic Use Only",
            "This system is developed solely for academic and research purposes. It is not intended for real-world medical diagnosis or treatment. Always consult a qualified healthcare professional before making medical decisions.",
        ))

        # ── Patient information ──────────────────────────────────────────
        elements.extend(fmt.format_patient_info(report_data.get("patient")))

        # ── Risk summary (patient report) ────────────────────────────────
        elements.extend(fmt.format_risk_summary(
            report_data.get("risk_level"),
            report_data.get("risk_percentage"),
        ))

        # ── Patient-specific body sections ───────────────────────────────
        elements.extend(fmt.section("Summary", report_data.get("summary")))
        elements.extend(fmt.section("Details", report_data.get("details")))
        elements.extend(fmt.section(
            "Lifestyle Recommendations",
            report_data.get("lifestyle_recommendations"),
        ))
        elements.extend(fmt.section(
            "Follow-Up Advice", report_data.get("follow_up_advice")
        ))

        # ── Doctor report sections ───────────────────────────────────────
        elements.extend(fmt.section(
            "Final Prediction", report_data.get("final_prediction")
        ))
        elements.extend(fmt.format_clinical_analysis(
            report_data.get("clinical_analysis")
        ))
        elements.extend(fmt.format_ecg_echo(
            report_data.get("ecg_analysis"), "ECG Analysis"
        ))
        elements.extend(fmt.format_ecg_echo(
            report_data.get("echo_analysis"), "Echocardiography Analysis"
        ))
        elements.extend(fmt.format_digital_twin(
            report_data.get("digital_twin")
        ))
        elements.extend(fmt.format_ai_recommendation(
            report_data.get("ai_recommendation")
        ))
        elements.extend(fmt.format_medical_explanation(
            report_data.get("medical_explanation")
        ))

        # ── Supporting references (doctor) ───────────────────────────────
        elements.extend(fmt.format_references(
            report_data.get("supporting_references")
        ))

        # ── Footer ───────────────────────────────────────────────────────
        elements.extend(fmt.format_footer())

        document.build(elements)
        buffer.seek(0)
        return buffer
