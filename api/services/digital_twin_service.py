from digital_twin.engine.twin_engine import TwinEngine
from digital_twin.models.patient_profile import PatientProfile
from digital_twin.simulation.simulator import TwinSimulator

from api.utils.logger import get_logger


class DigitalTwinService:
    """
    Service wrapper around the Digital Twin engine.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.engine = TwinEngine()

    def simulate_future(
        self,
        patient_data: dict,
        prediction_result: dict,
    ) -> dict:
        """
        Run Digital Twin simulation.
        """

        try:
            fusion = prediction_result.get("fusion", {})

            height = patient_data.get("height", 0)
            weight = patient_data.get("weight", 0)

            bmi = 0
            if height > 0:
                bmi = weight / ((height / 100) ** 2)

            profile = PatientProfile(
                patient_id=str(patient_data.get("patient_id", "unknown")),
                age=int(patient_data.get("age", 0)),
                gender=str(patient_data.get("gender", "Unknown")),
                systolic_bp=float(patient_data.get("ap_hi", 120)),
                diastolic_bp=float(patient_data.get("ap_lo", 80)),
                # The existing clinical request represents these as ordinal
                # categories; preserve those values without inventing units.
                cholesterol=float(patient_data.get("cholesterol", 1)),
                glucose=float(patient_data.get("gluc", 1)),
                bmi=round(bmi, 2),
                weight=float(weight or 0),
                smoking_status=bool(patient_data.get("smoke", 0)),
                alcohol_consumption=bool(patient_data.get("alco", 0)),
                exercise_level="Moderate",
                fusion_risk_percentage=float(fusion.get("risk_percentage", 0)),
            )

            self.logger.info("Running Digital Twin simulation")
            baseline_score = self.engine.calculate_risk(profile)
            scenarios = TwinSimulator().run_all(profile)
            return {
                "status": "success",
                "baseline_risk": round(baseline_score / 100, 4),
                "simulations": [
                    {
                        "scenario": scenario["scenario"],
                        "risk": round(scenario["risk_after"] / 100, 4),
                        "change": round(baseline_score - scenario["risk_after"], 2),
                    }
                    for scenario in scenarios
                ],
            }

        except Exception:
            self.logger.exception("Digital Twin simulation failed")

            return {
                "status": "error",
                "error": "Digital Twin simulation failed.",
            }
