import os
from typing import Optional

from rag.generator.generator import build_chat_prompt, call_groq_api, GROQ_MODEL
from rag.pipeline.rag_pipeline import RAGPipeline
from api.utils.logger import get_logger


class RAGService:
    """
    Service wrapper around the RAG pipeline.

    Responsibilities:
    - get_explanation(): used internally by PredictionService during /predict.
    - ask(): used by the public AI Health Assistant chatbot (POST /rag/ask).
    """

    _shared_pipeline = None

    def __init__(self):
        self.logger = get_logger(__name__)
        self.pipeline = None
        self.retriever = None

    def _get_pipeline(self) -> RAGPipeline:
        if RAGService._shared_pipeline is None:
            RAGService._shared_pipeline = RAGPipeline()

        self.pipeline = RAGService._shared_pipeline
        self.retriever = self.pipeline.retriever
        return self.pipeline

    def get_explanation(self, prediction_result: dict) -> dict:
        """
        Generate a structured medical explanation for a completed prediction.
        Called by PredictionService — not part of the chatbot flow.
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

            return self._get_pipeline().run(rag_input)

        except Exception as e:
            self.logger.exception("RAG pipeline failed")

            return {
                "status": "error",
                "error": str(e),
            }

    def ask(self, question: str, context: Optional[dict] = None) -> dict:
        """
        Answer a free-text medical question using RAG (AI Health Assistant).

        Flow:
          1. Build a FAISS search query from the question (augmented with
             optional prediction context if provided).
          2. Retrieve the top-k relevant guideline chunks.
          3. Build a conversational prompt (build_chat_prompt).
          4. Call the Groq LLM (call_groq_api).
          5. Return the plain-text answer and the source chunks.

        Args:
            question: Free-text medical question from the user.
            context:  Optional dict with keys risk_level, risk_percentage,
                      ecg_class, ef_value. When present, the search query is
                      augmented with the clinical signals for a richer retrieval.

        Returns:
            dict with keys: answer (str), sources (list[dict]), status (str).
        """
        try:
            self.logger.info("RAG chatbot: question received")

            # ── Step 1: Build retrieval query ──────────────────────────────
            # Start with the raw user question. If prediction context is
            # available, append clinical terms to improve retrieval accuracy.
            query = question
            if context:
                risk_level = context.get("risk_level")
                ecg_class = context.get("ecg_class")
                ef_value = context.get("ef_value")

                if risk_level:
                    query += f" {risk_level} cardiovascular risk"
                if ecg_class and ecg_class != "NORM":
                    query += f" {ecg_class} ECG findings"
                if ef_value is not None and ef_value < 55:
                    query += " reduced ejection fraction"

            # ── Step 2: Retrieve relevant guideline chunks ─────────────────
            chunks = self._get_pipeline().retriever.retrieve(query, top_k=4)
            self.logger.info("RAG chatbot: retrieved %d chunks", len(chunks))

            # ── Step 3 & 4: Build prompt and call LLM ─────────────────────
            prompt = build_chat_prompt(question, chunks)
            raw_answer = call_groq_api(prompt)

            # ── Step 5: Shape the response ─────────────────────────────────
            sources = [
                {
                    "text": c["text"][:200],
                    "source": c["source"],
                    "page": c["page"],
                    "category": c["category"],
                    "relevance_score": round(c["score"], 4),
                }
                for c in chunks
            ]

            self.logger.info("RAG chatbot: answer generated successfully")

            return {
                "answer": raw_answer.strip(),
                "sources": sources,
                "status": "success",
            }

        except Exception as e:
            self.logger.exception("RAG chatbot: ask() failed")
            return {
                "answer": "I'm sorry, I was unable to answer your question at this time.",
                "sources": [],
                "status": "error",
            }

    def health(self) -> dict:
        """
        Return readiness information about the RAG subsystem.
        Used by GET /rag/health.
        """
        try:
            if self.retriever is None:
                return {
                    "status": "unavailable",
                    "vector_store_loaded": False,
                    "chunks_indexed": 0,
                    "llm_model": GROQ_MODEL,
                    "groq_api_configured": bool(os.environ.get("GROQ_API_KEY", "")),
                }

            chunks_indexed = len(self.retriever.chunks)
            vector_store_loaded = chunks_indexed > 0
        except Exception:
            chunks_indexed = 0
            vector_store_loaded = False

        groq_api_configured = bool(os.environ.get("GROQ_API_KEY", ""))

        return {
            "status": "ok" if (vector_store_loaded and groq_api_configured) else "unavailable",
            "vector_store_loaded": vector_store_loaded,
            "chunks_indexed": chunks_indexed,
            "llm_model": GROQ_MODEL,
            "groq_api_configured": groq_api_configured,
        }
