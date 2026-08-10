def generate_recommendations(patient):

    recommendations = []

    systolic_bp = patient.systolic_bp or 120
    cholesterol = patient.cholesterol or 200
    glucose = patient.glucose or 100
    bmi = patient.bmi or 25
    smoking_status = patient.smoking_status or False
    exercise_level = patient.exercise_level or "Moderate"

    if systolic_bp > 140:

        recommendations.append(

            "Blood pressure management advised"

        )

    if cholesterol > 240:

        recommendations.append(

            "Reduce cholesterol levels"

        )

    if glucose > 140:

        recommendations.append(

            "Improve diabetes control"

        )

    if bmi > 30:

        recommendations.append(

            "Weight reduction recommended"

        )

    if smoking_status:

        recommendations.append(

            "Smoking cessation advised"

        )

    if exercise_level == "Low":

        recommendations.append(

            "Increase physical activity"

        )

    return recommendations