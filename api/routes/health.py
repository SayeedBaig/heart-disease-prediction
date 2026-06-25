from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "backend": "CardioAI",
        "version": "2.0.0"
    }