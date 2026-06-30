from contextlib import asynccontextmanager
from api.routes.reports import router as reports_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.predict import router as predict_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting CardioAI Backend...")

    # Future:
    # Load SystemPipeline here
    # Load database connection
    # Load RAG resources

    yield

    print("Shutting down CardioAI Backend...")


app = FastAPI(
    title="CardioAI API",
    description="Multi-Modal Heart Disease Prediction Backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(reports_router)

@app.get("/")
def root():
    return {
        "message": "CardioAI Backend Running"
    }