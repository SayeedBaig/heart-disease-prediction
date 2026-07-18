from digital_twin.models.patient_profile import PatientProfile

from digital_twin.simulation.simulator import TwinSimulator

from digital_twin.recommendations.comparison import compare

from digital_twin.recommendations.summary import generate_summary


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

results = simulator.run_all(patient)

comparison = compare(

    patient.fusion_risk_percentage,

    results

)

summary = generate_summary(

    results

)

print()

print(results)

print()

print(comparison)

print()

print(summary)
