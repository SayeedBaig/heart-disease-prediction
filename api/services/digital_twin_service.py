from digital_twin.engine.twin_engine import TwinEngine

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

            twin_input = {
                "age": patient_data.get("age", 0),
                "bp": patient_data.get("ap_hi", 0),
                "cholesterol": patient_data.get("cholesterol", 0),
                "bmi": round(bmi, 2),
                "smoking": patient_data.get("smoke", 0),
                "risk_score": fusion.get("risk_percentage", 0) / 100,
            }

            self.logger.info("Running Digital Twin simulation")

            return self.engine.run(twin_input)

        except Exception as e:
            self.logger.exception("Digital Twin simulation failed")

            return {
                "status": "error",
                "error": str(e),
            }