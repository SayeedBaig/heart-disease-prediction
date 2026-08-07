"""
Food & Lifestyle Recommendation Rules Engine
Author: Akash
Module 4 — Food Recommendation Engine

Purpose:
    Deterministic, rule-based recommendations for food, water,
    exercise, sleep, and lifestyle based on prediction risk level
    and patient profile. Rule-based (not LLM-generated) because
    dietary/medical guidance must be reliable and consistent.
"""

BASE_RECOMMENDATIONS = {
    "High": {
        "recommended_foods": [
            "Leafy green vegetables", "Oats and whole grains",
            "Berries", "Fatty fish (salmon, mackerel)", "Nuts (unsalted)"
        ],
        "avoid_foods": [
            "Fried foods", "Processed meats", "Sugary drinks",
            "High-sodium packaged snacks", "Excess red meat"
        ],
        "water_intake_liters": 2.5,
        "exercise": "Light activity only, as approved by a doctor — "
                    "e.g. short daily walks. Avoid strenuous exercise "
                    "without medical clearance.",
        "sleep_hours": "7-9 hours, consistent schedule",
        "lifestyle": [
            "Quit smoking immediately if applicable",
            "Limit alcohol intake",
            "Monitor blood pressure regularly",
            "Reduce stress through relaxation techniques"
        ]
    },
    "Medium": {
        "recommended_foods": [
            "Leafy green vegetables", "Whole grains", "Fruits",
            "Lean protein (chicken, fish)", "Legumes"
        ],
        "avoid_foods": [
            "Fried foods", "Sugary drinks", "Processed snacks",
            "Excess salt"
        ],
        "water_intake_liters": 2.5,
        "exercise": "Moderate activity — 150 minutes/week of brisk "
                    "walking, cycling, or swimming.",
        "sleep_hours": "7-9 hours, consistent schedule",
        "lifestyle": [
            "Reduce smoking and alcohol",
            "Manage stress",
            "Routine health checkups"
        ]
    },
    "Low": {
        "recommended_foods": [
            "Balanced diet with vegetables, fruits, whole grains",
            "Lean protein", "Healthy fats (olive oil, nuts, avocado)"
        ],
        "avoid_foods": [
            "Excess processed food", "Excess sugar"
        ],
        "water_intake_liters": 2.0,
        "exercise": "150 minutes/week moderate activity, or 75 "
                    "minutes/week vigorous activity.",
        "sleep_hours": "7-9 hours",
        "lifestyle": [
            "Maintain healthy habits",
            "Routine annual checkups"
        ]
    }
}


def get_recommendations(risk_level: str, patient_profile: dict = None) -> dict:
    """
    Returns food/lifestyle recommendations based on risk level,
    with adjustments based on patient profile flags.

    Args:
        risk_level: "High", "Medium", or "Low"
        patient_profile: optional dict, e.g.
            {"diabetes": True, "smoker": True, "age": 55}
    """
    risk_level = risk_level.capitalize() if risk_level else "Medium"

    if risk_level not in BASE_RECOMMENDATIONS:
        print(f"  WARNING: Unknown risk level '{risk_level}', defaulting to Medium")
        risk_level = "Medium"

    # Start from a copy so we don't mutate the base dict
    recommendations = {
        "recommended_foods": list(BASE_RECOMMENDATIONS[risk_level]["recommended_foods"]),
        "avoid_foods": list(BASE_RECOMMENDATIONS[risk_level]["avoid_foods"]),
        "water_intake_liters": BASE_RECOMMENDATIONS[risk_level]["water_intake_liters"],
        "exercise": BASE_RECOMMENDATIONS[risk_level]["exercise"],
        "sleep_hours": BASE_RECOMMENDATIONS[risk_level]["sleep_hours"],
        "lifestyle": list(BASE_RECOMMENDATIONS[risk_level]["lifestyle"])
    }

    # Profile-based adjustments
    if patient_profile:
        if patient_profile.get("diabetes"):
            recommendations["avoid_foods"].append("Refined sugar and white bread")
            recommendations["recommended_foods"].append("Low-glycemic-index foods")

        if patient_profile.get("smoker"):
            if "Quit smoking immediately if applicable" not in recommendations["lifestyle"]:
                recommendations["lifestyle"].insert(0, "Quit smoking — highest priority action")

        if patient_profile.get("age") and patient_profile["age"] >= 65:
            recommendations["exercise"] = (
                "Gentle activity only — short walks, chair exercises. "
                "Consult doctor before starting any new routine."
            )

    return {
        "risk_level": risk_level,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    print("=== Food Recommendation Rules Engine Test ===\n")

    print("Test 1: High risk, no profile")
    result = get_recommendations("High")
    print(f"Recommended: {result['recommendations']['recommended_foods']}")
    print(f"Avoid: {result['recommendations']['avoid_foods']}")
    print(f"Exercise: {result['recommendations']['exercise']}\n")

    print("Test 2: Medium risk, diabetic + smoker profile")
    profile = {"diabetes": True, "smoker": True, "age": 45}
    result = get_recommendations("Medium", profile)
    print(f"Avoid: {result['recommendations']['avoid_foods']}")
    print(f"Lifestyle: {result['recommendations']['lifestyle']}\n")

    print("Test 3: Low risk, elderly profile")
    profile = {"age": 70}
    result = get_recommendations("Low", profile)
    print(f"Exercise: {result['recommendations']['exercise']}")