from typing import Dict, Any


class PatientReport:
    """
    Generates a simplified report for patients.
    """

    def generate(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:

        fusion = prediction.get("fusion", {})

        rag = explanation.get(
            "explanation",
            {}
        )

        recommendations = []

        for item in rag.get(
            "recommendations",
            []
        ):
            recommendations.append(
                item.get("text")
            )

        return {

            "report_type": "patient",

            "risk_level": fusion.get(
                "final_level"
            ),

            "risk_percentage": fusion.get(
                "risk_percentage"
            ),

            "summary": rag.get(
                "summary"
            ),

            "details": rag.get(
                "details"
            ),

            "lifestyle_recommendations": rag.get(
                "lifestyle_suggestions",
                []
            ),

            "follow_up_advice": recommendations,

            "generated_by": "CardioAI",

            "version": "2.0",
        }