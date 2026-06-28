class RAGService:
    """
    Interface between backend API and
    the RAG implementation.

    Actual retrieval logic will be
    implemented by the RAG module.
    """

    def get_explanation(self, prediction_result: dict) -> dict:
        """
        Placeholder for future RAG explanation.
        """

        return {
            "status": "pending",
            "message": "RAG explanation not implemented yet.",
            "explanation": None,
        }