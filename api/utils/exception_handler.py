from fastapi import HTTPException

from api.utils.logger import get_logger

logger = get_logger(__name__)


def handle_prediction_exception(exception: Exception) -> None:
    """
    Log the real exception internally and raise a safe HTTP 500
    that never exposes internal details to API clients.
    """
    logger.exception("Prediction pipeline error: %s", exception)

    raise HTTPException(
        status_code=500,
        detail="An internal error occurred. Please try again.",
    )