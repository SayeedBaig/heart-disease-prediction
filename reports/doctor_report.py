from typing import Dict, Any


class DoctorReport:
    """
    Generates a detailed report for doctors.
    """

    def generate(
        self,
        prediction: Dict[str, Any],
        explanation: Dict[str, Any],
        digital_twin: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {

            "report_type": "doctor",

            "final_prediction": prediction.get("fusion", {}),

            "clinical_analysis": prediction.get("clinical", {}),

            "ecg_analysis": prediction.get("ecg", {}),

            "echo_analysis": prediction.get("echo", {}),

            "ai_recommendation": prediction.get("rag", {}),

            "medical_explanation": explanation.get(
                "explanation",
                {}
            ),

            "supporting_references": explanation.get(
                "chunks",
                []
            ),

            "digital_twin": digital_twin,

            "generated_by": "CardioAI",

            "version": "2.0",
        }