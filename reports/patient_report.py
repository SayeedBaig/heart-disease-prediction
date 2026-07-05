from typing import Any, Dict, Optional


class PatientReport:
    """
    Generates a simplified report for patients.
    """

    def generate(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        patient: Optional[Dict[str, Any]] = None,
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

            "patient": patient,

            "generated_by": "CardioAI",

            "version": "2.0",
        }