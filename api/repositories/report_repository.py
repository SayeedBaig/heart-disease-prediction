from sqlalchemy.orm import Session

from api.models.report import Report


class ReportRepository:

    def __init__(self, db: Session):
        self.db = db

    def save(
        self,
        patient_id: int,
        prediction_id: int,
        report_type: str,
        report_format: str,
    ) -> Report:

        report = Report(
            patient_id=patient_id,
            prediction_id=prediction_id,
            report_type=report_type,
            report_format=report_format,
        )

        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)

        return report