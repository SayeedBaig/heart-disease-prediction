"""
Food Recommendation Service
Author: Akash
Module 4 — Food Recommendation Engine

Purpose:
    Entry point for the food/lifestyle recommendation module.
    Wraps the rules engine and formats the final response.
"""

from food_recommendation.rules_engine import get_recommendations


class FoodRecommendationService:
    """
    Handles food/lifestyle recommendation requests.
    Called by: Patient Assistant (Module 3), Reports, Digital Twin.
    """

    def get_recommendations(self, risk_level: str, patient_profile: dict = None) -> dict:
        try:
            result = get_recommendations(risk_level, patient_profile)

            return {
                "status": "success",
                "risk_level": result["risk_level"],
                "recommended_foods": result["recommendations"]["recommended_foods"],
                "avoid_foods": result["recommendations"]["avoid_foods"],
                "water_intake_liters": result["recommendations"]["water_intake_liters"],
                "exercise": result["recommendations"]["exercise"],
                "sleep_hours": result["recommendations"]["sleep_hours"],
                "lifestyle": result["recommendations"]["lifestyle"]
            }

        except Exception as e:
            print(f"Food Recommendation Service error: {e}")
            return {
                "status": "error",
                "risk_level": risk_level,
                "recommended_foods": [],
                "avoid_foods": [],
                "water_intake_liters": None,
                "exercise": "",
                "sleep_hours": "",
                "lifestyle": []
            }


if __name__ == "__main__":
    print("=== Food Recommendation Service Test ===\n")

    service = FoodRecommendationService()

    print("Test: High risk, diabetic profile")
    result = service.get_recommendations("High", {"diabetes": True})
    print(f"Status: {result['status']}")
    print(f"Recommended: {result['recommended_foods']}")
    print(f"Avoid: {result['avoid_foods']}")
    print(f"Water: {result['water_intake_liters']}L")
    print(f"Exercise: {result['exercise']}")
    print(f"Sleep: {result['sleep_hours']}")
    print(f"Lifestyle: {result['lifestyle']}")