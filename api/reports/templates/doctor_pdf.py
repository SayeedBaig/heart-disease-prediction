from io import BytesIO

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate


class DoctorPDFTemplate:

    def generate(self, report_data: dict):
        buffer = BytesIO()

        document = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()

        elements = []

        elements.append(
            Paragraph("<b>CardioAI Doctor Report</b>", styles["Title"])
        )

        elements.append(
            Paragraph("<br/>", styles["Normal"])
        )

        for key, value in report_data.items():

            elements.append(
                Paragraph(
                    f"<b>{key.replace('_', ' ').title()}</b>",
                    styles["Heading2"],
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