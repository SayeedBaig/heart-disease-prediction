from digital_twin.engine.twin_engine import TwinEngine

from digital_twin.evaluation.edge_cases import patients

from digital_twin.recommendations.recommendation_engine import (
    generate_recommendations
)


engine = TwinEngine()

print("\nDIGITAL TWIN VALIDATION")
print("=" * 40)

for patient in patients:

    risk = engine.calculate_risk(patient)

    confidence = engine.calculate_confidence(patient)

    recommendations = generate_recommendations(patient)

    print()

    print("Patient ID:", patient.patient_id)

    print("Risk Score:", risk)

    print("Confidence:", confidence)

    print()

    print("Recommendations:")

    for recommendation in recommendations:

        print("-", recommendation)

    print("-" * 40)