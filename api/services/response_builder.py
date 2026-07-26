class ResponseBuilder:
    """
    Builds the final API response.

    Combines prediction output with
    future RAG explanation and
    Digital Twin simulation.
    """

    def build(
        self,
        prediction: dict,
        explanation: dict | None = None,
        digital_twin: dict | None = None,
    ) -> dict:

        return {
            "prediction": prediction,
            "explanation": explanation or {
                "status": "pending",
                "message": "RAG explanation not available yet."
            },
            "digital_twin": digital_twin or {
                "status": "pending",
                "message": "Digital Twin simulation not available yet."
            }
        }