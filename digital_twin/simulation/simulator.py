from digital_twin.engine.twin_engine import TwinEngine

from digital_twin.scenarios.medication import simulate as medication
from digital_twin.scenarios.exercise import simulate as exercise
from digital_twin.scenarios.diabetes import simulate as diabetes
from digital_twin.scenarios.lifestyle import simulate as lifestyle

from digital_twin.scenarios.smoking_scenario import simulate as smoking
from digital_twin.scenarios.weight_scenario import simulate as weight
from digital_twin.scenarios.bp_scenario import simulate as bp
from digital_twin.scenarios.cholesterol_scenario import simulate as cholesterol

from digital_twin.scenario_engine.manager import ScenarioManager
from digital_twin.timeline.projection import TimelineProjection
from digital_twin.risk_projection.meter import RiskImprovementMeter


class TwinSimulator:

    def __init__(self):

        self.engine = TwinEngine()

        self.timeline = TimelineProjection()

        self.scenario_manager = ScenarioManager()

        self.meter = RiskImprovementMeter()

    # ---------------------------------------------------
    # Existing Simulation
    # ---------------------------------------------------

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

                "confidence":
                self.engine.calculate_confidence(
                    updated_patient
                ),

                "bp":
                f"{updated_patient.systolic_bp}/{updated_patient.diastolic_bp}",

                "cholesterol":
                updated_patient.cholesterol,

                "glucose":
                updated_patient.glucose,

                "bmi":
                updated_patient.bmi

            })

        return results

    # ---------------------------------------------------
    # Interactive Simulation
    # ---------------------------------------------------

    def simulate_custom(
        self,
        patient,
        changes
    ):

        return self.engine.simulate_custom(
            patient,
            changes
        )

    # ---------------------------------------------------
    # Scenario Engine
    # ---------------------------------------------------

    def save_scenario(
        self,
        name,
        patient
    ):

        self.scenario_manager.save_scenario(
            name,
            patient
        )

    def compare_saved_scenarios(self):

        return self.scenario_manager.simulate_all()

    # ---------------------------------------------------
    # Timeline Projection
    # ---------------------------------------------------

    def project_timeline(
        self,
        patient,
        visits=5
    ):

        return self.timeline.project(
            patient,
            visits
        )

    # ---------------------------------------------------
    # Risk Improvement Meter
    # ---------------------------------------------------

    def calculate_improvement(
        self,
        current_risk,
        projected_risk
    ):

        return self.meter.calculate(

            current_risk,

            projected_risk

        )

    # ---------------------------------------------------
    # Compare Original vs Modified
    # ---------------------------------------------------

    def compare_patients(
        self,
        original,
        modified
    ):

        return self.engine.compare_patients(

            original,

            modified

        )