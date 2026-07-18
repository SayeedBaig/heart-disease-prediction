from digital_twin.engine.twin_engine import TwinEngine

from digital_twin.scenarios.medication import simulate as medication
from digital_twin.scenarios.exercise import simulate as exercise
from digital_twin.scenarios.diabetes import simulate as diabetes
from digital_twin.scenarios.lifestyle import simulate as lifestyle

from digital_twin.scenarios.smoking_scenario import simulate as smoking
from digital_twin.scenarios.weight_scenario import simulate as weight
from digital_twin.scenarios.bp_scenario import simulate as bp
from digital_twin.scenarios.cholesterol_scenario import simulate as cholesterol


class TwinSimulator:

    def __init__(self):

        self.engine = TwinEngine()

    def run_all(self, patient):

        results = []

        scenarios = {

            "Smoking Cessation": smoking,

            "Weight Reduction": weight,

            "BP Control": bp,

            "Cholesterol Control": cholesterol,

            "Exercise Improvement": exercise,

            "Medication Adherence": medication,

            "Diabetes Control": diabetes,

            "Combined Lifestyle": lifestyle

        }

        for name, func in scenarios.items():

            updated_patient = func(patient)

            projected_risk = self.engine.calculate_risk(
                updated_patient
            )

            results.append({

                "scenario": name,

                "risk_after": projected_risk,

                "bp": updated_patient.systolic_bp,

                "cholesterol": updated_patient.cholesterol,

                "glucose": updated_patient.glucose,

                "bmi": updated_patient.bmi

            })

        return results
