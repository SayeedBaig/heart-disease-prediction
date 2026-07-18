from digital_twin.models.patient_profile import PatientProfile

from digital_twin.simulation.simulator import TwinSimulator

from digital_twin.recommendations.comparison import compare
from digital_twin.recommendations.summary import generate_summary


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


patient = PatientProfile(

    patient_id="001",

    age=55,

    gender="Male",

    systolic_bp=150,

    diastolic_bp=95,

    cholesterol=270,

    glucose=170,

    bmi=31,

    weight=88,

    smoking_status=True,

    fusion_risk_percentage=78

)

simulator = TwinSimulator()

# ---------------------------------------------------
# 1. Existing Scenario Simulation
# ---------------------------------------------------

print_header("ALL PREDEFINED SCENARIOS")

results = simulator.run_all(patient)

for result in results:
    print(result)

# ---------------------------------------------------
# 2. Comparison
# ---------------------------------------------------

print_header("BEST SCENARIO")

comparison = compare(
    patient.fusion_risk_percentage,
    results
)

print(comparison)

# ---------------------------------------------------
# 3. Summary
# ---------------------------------------------------

print_header("SUMMARY")

summary = generate_summary(results)

print(summary)

# ---------------------------------------------------
# 4. Interactive Simulation
# ---------------------------------------------------

print_header("CUSTOM SIMULATION")

custom = simulator.simulate_custom(

    patient,

    {

        "systolic_bp": 120,

        "diastolic_bp": 80,

        "cholesterol": 180,

        "glucose": 110,

        "bmi": 25,

        "smoking_status": False,

        "exercise_level": "High"

    }

)

print(custom)

# ---------------------------------------------------
# 5. Compare Patients
# ---------------------------------------------------

print_header("PATIENT COMPARISON")

comparison_result = simulator.compare_patients(

    patient,

    custom["updated_patient"]

)

print(comparison_result)

# ---------------------------------------------------
# 6. Risk Improvement Meter
# ---------------------------------------------------

print_header("RISK IMPROVEMENT")

meter = simulator.calculate_improvement(

    patient.fusion_risk_percentage,

    custom["risk"]

)

print(meter)

# ---------------------------------------------------
# 7. Timeline Projection
# ---------------------------------------------------

print_header("TIMELINE PROJECTION")

timeline = simulator.project_timeline(

    patient,

    visits=5

)

for visit in timeline:
    print(visit)

# ---------------------------------------------------
# 8. Scenario Manager
# ---------------------------------------------------

print_header("SCENARIO MANAGER")

simulator.save_scenario(

    "Current",

    patient

)

simulator.save_scenario(

    "Improved Lifestyle",

    custom["updated_patient"]

)

saved = simulator.compare_saved_scenarios()

for scenario in saved:
    print(scenario)

print("\nDigital Twin Phase-2 Test Completed Successfully.")