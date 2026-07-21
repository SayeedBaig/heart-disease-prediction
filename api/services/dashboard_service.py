from typing import List

from api.repositories.dashboard_repository import DashboardRepository
from api.schemas.dashboard import ActivityItem

class DashboardService:
    def __init__(self, repository: DashboardRepository):
        self.repository = repository

    def get_recent_activity(self, limit: int = 20) -> List[ActivityItem]:
        activities = []

        # Patients
        patients = self.repository.get_latest_patients(limit)
        for p in patients:
            activities.append(ActivityItem(
                type="PATIENT_REGISTERED",
                title="New Patient Registered",
                description=f"Patient {p.full_name} ({p.patient_id}) was registered.",
                timestamp=p.created_at.isoformat() + "Z" if p.created_at else "",
                resource_id=str(p.patient_id)
            ))

        # Appointments
        appointments = self.repository.get_latest_appointments(limit)
        for a in appointments:
            patient_name = a.patient.full_name if a.patient else "Unknown Patient"
            activities.append(ActivityItem(
                type="APPOINTMENT_BOOKED",
                title="Appointment Booked",
                description=f"An appointment was booked for {patient_name}.",
                timestamp=a.created_at.isoformat() + "Z" if a.created_at else "",
                resource_id=str(a.appointment_id)
            ))

        # Predictions
        predictions = self.repository.get_latest_predictions(limit)
        for p in predictions:
            patient_name = p.patient.full_name if p.patient else "Unknown Patient"
            activities.append(ActivityItem(
                type="PREDICTION_COMPLETED",
                title="Prediction Completed",
                description=f"Heart disease prediction completed for {patient_name}. Risk Level: {p.risk_level}.",
                timestamp=p.created_at.isoformat() + "Z" if p.created_at else "",
                resource_id=str(p.id)
            ))

        # Reports
        reports = self.repository.get_latest_reports(limit)
        for r in reports:
            patient_name = r.patient.full_name if r.patient else "Unknown Patient"
            activities.append(ActivityItem(
                type="REPORT_GENERATED",
                title="Report Generated",
                description=f"A {r.report_type} report was generated for {patient_name}.",
                timestamp=r.generated_at.isoformat() + "Z" if r.generated_at else "",
                resource_id=str(r.id)
            ))

        # Doctor Notes
        notes = self.repository.get_latest_doctor_notes(limit)
        for n in notes:
            patient_name = "Unknown Patient"
            if n.diagnosis and n.diagnosis.patient:
                patient_name = n.diagnosis.patient.full_name
            activities.append(ActivityItem(
                type="DOCTOR_NOTE_ADDED",
                title="Doctor Note Added",
                description=f"A new note was added for {patient_name}.",
                timestamp=n.created_at.isoformat() + "Z" if n.created_at else "",
                resource_id=str(n.note_id)
            ))

        # Sort by timestamp descending
        activities.sort(key=lambda x: x.timestamp, reverse=True)

        return activities[:limit]
