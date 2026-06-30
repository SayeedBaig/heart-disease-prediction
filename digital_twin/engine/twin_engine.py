from digital_twin.models.patient_profile import PatientProfile


class TwinEngine:

    def calculate_risk(self, patient: PatientProfile):

        risk = patient.fusion_risk_percentage

        if patient.systolic_bp < 140:
            risk -= 4

        if patient.cholesterol < 240:
            risk -= 4

        if patient.glucose < 140:
            risk -= 3

        if patient.bmi < 30:
            risk -= 3

        if not patient.smoking_status:
            risk -= 3

        if patient.exercise_level == "High":
            risk -= 5

        risk = max(0, risk)

        return round(risk, 2)
