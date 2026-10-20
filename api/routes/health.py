from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from api.config.upload_config import PROJECT_ROOT
from api.database.connection import engine
from rag.model_config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "backend": "CardioAI",
        "version": "2.0.0"
    }


@router.get("/ready")
def readiness_check():
    """Check database connectivity and required tracked runtime artifacts."""

    required_assets = (
        "clinical_module/agent/stack_model.pkl",
        "clinical_module/agent/scaler.pkl",
        "ecg_module/models/ecg_attention_calibrated.pth",
        "echo_module/models/echo_best_model.keras",
        "rag/vector_store/index.faiss",
        "rag/vector_store/metadata.pkl",
    )
    assets_ready = all((PROJECT_ROOT / path).is_file() for path in required_assets)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=EMBEDDING_MODEL,
            revision=EMBEDDING_MODEL_REVISION,
            local_files_only=True,
        )
    except Exception:
        assets_ready = False

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_ready = True
    except Exception:
        database_ready = False

    if not (database_ready and assets_ready):
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    return {"status": "ready"}
