from io import BytesIO
from typing import Any, Dict

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate


class PdfRenderer:
    def render(self, report_data: Dict[str, Any]) -> BytesIO:
        buffer = BytesIO()

        document = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()

        elements = []

        title = report_data.get(
            "report_type",
            "CardioAI Report"
        ).title()

        elements.append(
            Paragraph(
                f"<b>CardioAI {title} Report</b>",
                styles["Title"],
            )
        )

        patient = report_data.get("patient")

        if patient:
            elements.append(
                Paragraph("<br/><b>Patient Information</b>", styles["Heading2"])
            )

            for key, value in patient.items():
                elements.append(
                    Paragraph(
                        f"<b>{key.replace('_', ' ').title()}:</b> {value}",
                        styles["BodyText"],
                    )
                )

        elements.append(
            Paragraph("<br/><b>Report Details</b>", styles["Heading2"])
        )

        for key, value in report_data.items():

            if key == "patient":
                continue

            elements.append(
                Paragraph(
                    f"<b>{key.replace('_', ' ').title()}</b>",
                    styles["Heading3"],
                )
            )

            elements.append(
                Paragraph(
                    str(value),
                    styles["BodyText"],
                )
            )

        document.build(elements)

        buffer.seek(0)

        return buffer