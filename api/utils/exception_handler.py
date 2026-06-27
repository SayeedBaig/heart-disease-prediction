from fastapi import HTTPException


def handle_prediction_exception(exception: Exception):
    """
    Convert backend exceptions into
    standardized HTTP responses.
    """

    raise HTTPException(
        status_code=500,
        detail=str(exception),
    )