from typing import List, Optional

from pydantic import BaseModel, Field


class ChatContext(BaseModel):
    """
    Optional structured prediction context to enrich the chatbot answer.
    When provided, the retriever builds an augmented query that combines
    the user's question with the clinical signals.
    """

    risk_level: Optional[str] = Field(
        default=None,
        description="Fused cardiovascular risk level: 'Low', 'Medium', or 'High'.",
    )
    risk_percentage: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Fused risk percentage (0–100).",
    )
    ecg_class: Optional[str] = Field(
        default=None,
        description="ECG classification: 'NORM', 'MI', 'STTC', 'CD', or 'HYP'.",
    )
    ef_value: Optional[float] = Field(
        default=None,
        description="Ejection fraction value from echocardiography.",
    )


class ChatAskRequest(BaseModel):
    """
    Request body for the POST /rag/ask chatbot endpoint.
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Free-text medical question from the user.",
        examples=["What are common symptoms of heart disease?"],
    )
    context: Optional[ChatContext] = Field(
        default=None,
        description=(
            "Optional prediction context. When supplied, the retriever augments "
            "the search query with clinical signals for a more targeted answer."
        ),
    )


class ChatSource(BaseModel):
    """
    A single medical guideline chunk that was used to ground the LLM answer.
    Returned so the frontend can render citation cards.
    """

    text: str = Field(description="Excerpt from the retrieved guideline chunk.")
    source: str = Field(description="Document filename (e.g. 'ACC_AHA_2019.pdf').")
    page: int = Field(description="Page number within the source document.")
    category: str = Field(description="Knowledge category (e.g. 'risk_factors').")
    relevance_score: float = Field(
        description="FAISS L2 distance — lower means more relevant."
    )


class ChatAskResponse(BaseModel):
    """
    Response body for POST /rag/ask.
    """

    answer: str = Field(description="LLM-generated answer grounded in retrieved guidelines.")
    sources: List[ChatSource] = Field(
        description="Guideline chunks used to generate the answer (for citation display)."
    )
    status: str = Field(description="'success' or 'error'.")


class RagHealthResponse(BaseModel):
    """
    Response body for GET /rag/health.
    """

    status: str = Field(description="'ok' or 'unavailable'.")
    vector_store_loaded: bool = Field(description="True if FAISS index is loaded.")
    chunks_indexed: int = Field(description="Total number of chunks in the vector store.")
    llm_model: str = Field(description="Active Groq LLM model identifier.")
    groq_api_configured: bool = Field(description="True if GROQ_API_KEY env var is set.")
