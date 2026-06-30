def generate_recommendations(patient):

    recommendations = []

    if patient.systolic_bp > 140:

        recommendations.append(

            "Blood pressure management advised"

        )

    if patient.cholesterol > 240:

        recommendations.append(

            "Reduce cholesterol levels"

        )

    if patient.glucose > 140:

        recommendations.append(

            "Improve diabetes control"

        )

    if patient.bmi > 30:

        recommendations.append(

            "Weight reduction recommended"

        )

    if patient.smoking_status:

        recommendations.append(

            "Smoking cessation advised"

        )

    if patient.exercise_level == "Low":

        recommendations.append(

            "Increase physical activity"

        )

    return recommendations