from contextlib import asynccontextmanager
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from fastapi import FastAPI


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

app.include_router(health_router)
app.include_router(predict_router)

@app.get("/")
def root():
    return {
        "message": "CardioAI Backend Running"
    }