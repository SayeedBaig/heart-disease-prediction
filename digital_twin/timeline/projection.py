from copy import deepcopy

from digital_twin.engine.twin_engine import TwinEngine


class TimelineProjection:
    """
    Simulates future patient risk over multiple visits.
    """

    def __init__(self):
        self.engine = TwinEngine()

    def project(self, patient, visits=5):

        projections = []

        future_patient = deepcopy(patient)

        for visit in range(1, visits + 1):

            # Simulate gradual improvements

            future_patient.systolic_bp = max(
                110,
                future_patient.systolic_bp - 3
            )

            future_patient.diastolic_bp = max(
                70,
                future_patient.diastolic_bp - 2
            )

            future_patient.cholesterol = max(
                150,
                future_patient.cholesterol - 5
            )

            future_patient.glucose = max(
                90,
                future_patient.glucose - 2
            )

            future_patient.bmi = max(
                20,
                future_patient.bmi - 0.3
            )

            risk = self.engine.calculate_risk(future_patient)

            projections.append({
                "visit": visit,
                "risk": risk,
                "bp": f"{future_patient.systolic_bp}/{future_patient.diastolic_bp}",
                "cholesterol": round(future_patient.cholesterol, 1),
                "glucose": round(future_patient.glucose, 1),
                "bmi": round(future_patient.bmi, 1)
            })

        return projections