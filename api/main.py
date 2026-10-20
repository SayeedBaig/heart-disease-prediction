import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.patient import router as patient_router
from api.routes.predict import router as predict_router
from api.routes.reports import router as reports_router
from api.routes.upload import router as upload_router
from api.routes.doctor import router as doctor_router
from api.routes.history import router as history_router
from api.routes.doctor_note import router as doctor_note_router
from api.services.upload_cleanup_service import UploadCleanupService
from api.utils.logger import get_logger
from api.routes.appointment import router as appointment_router
from api.routes.diagnosis import router as diagnosis_router
from api.routes.rag import router as rag_router
from api.routes.dashboard import router as dashboard_router
from api.routes.doctor_portal import router as doctor_portal_router
from api.routes.digital_twin import router as digital_twin_router


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CardioAI Backend...")

    if os.getenv("AUTO_CREATE_SCHEMA", "false").strip().lower() == "true":
        # Convenience for disposable local development only. Production schema
        # changes are applied explicitly with Alembic before service startup.
        from api.database.base import Base
        from api.database.connection import engine
        import api.models  # noqa: F401 — register all models before create_all

        Base.metadata.create_all(bind=engine)

    if os.getenv("CLEANUP_UPLOADS_ON_STARTUP", "false").strip().lower() == "true":
        cleanup_summary = UploadCleanupService().cleanup_all()
        logger.info("Upload Cleanup: %d file(s) removed.", cleanup_summary["total_deleted"])

    yield

    logger.info("Shutting down CardioAI Backend...")


app = FastAPI(
    title="CardioAI API",
    description="""
    AI-powered multi-modal heart disease prediction platform.

    Features:
    - Patient Registration
    - ECG & Echo Upload
    - Heart Disease Prediction
    - AI Medical Explanation (RAG)
    - Digital Twin Simulation
    - Prediction History
    - Doctor & Patient Reports
    - PDF Report Generation
    """,
    version="2.0.0",
    lifespan=lifespan,
)

from urllib.parse import urlsplit

_cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
if not _cors_origins or any("*" in origin for origin in _cors_origins):
    raise RuntimeError(
        "CORS_ORIGINS must contain one or more explicit comma-separated origins; "
        "wildcard origins are not allowed."
    )
for origin in _cors_origins:
    parsed_origin = urlsplit(origin)
    if (
        parsed_origin.scheme not in {"http", "https"}
        or not parsed_origin.netloc
        or parsed_origin.path
        or parsed_origin.query
        or parsed_origin.fragment
        or parsed_origin.username
        or parsed_origin.password
    ):
        raise RuntimeError(f"Invalid CORS origin {origin!r}; provide scheme and host only.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(reports_router)
app.include_router(upload_router)
app.include_router(patient_router)
app.include_router(doctor_router)
app.include_router(appointment_router)
app.include_router(diagnosis_router)
app.include_router(history_router)
app.include_router(doctor_note_router)
app.include_router(rag_router)
app.include_router(dashboard_router)
app.include_router(doctor_portal_router)
app.include_router(digital_twin_router)

@app.get("/")
def root():
    return {"message": "CardioAI Backend Running"}
