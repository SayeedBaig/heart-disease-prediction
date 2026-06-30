from rag.pipeline.rag_pipeline import RAGPipeline

from api.utils.logger import get_logger


class RAGService:
    """
    Service wrapper around the RAG pipeline.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.pipeline = RAGPipeline()

    def get_explanation(self, prediction_result: dict) -> dict:
        """
        Generate medical explanation using the RAG pipeline.
        """

        try:
            fusion = prediction_result.get("fusion", {})

            rag_input = {
                "risk_level": fusion.get("final_level"),
                "risk_percentage": int(fusion.get("risk_percentage", 0)),
                "ecg_class": prediction_result.get("ecg", {}).get("ecg_class"),
                "ef_value": prediction_result.get("echo", {}).get("ef_value"),
            }

            self.logger.info("Running RAG pipeline")

            return self.pipeline.run(rag_input)

        except Exception as e:
            self.logger.exception("RAG pipeline failed")

            return {
                "status": "error",
                "error": str(e),
            }