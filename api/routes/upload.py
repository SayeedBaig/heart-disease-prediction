from fastapi import APIRouter, Depends, File, UploadFile

from api.services.upload_service import UploadService
from api.utils.authorization import get_current_actor

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)

upload_service = UploadService()


@router.post(
    "/ecg",
    summary="Upload ECG file",
    description="Uploads an ECG file for heart disease prediction.",
    response_description="ECG uploaded successfully.",
)
def upload_ecg(
    file: UploadFile = File(...),
    _actor=Depends(get_current_actor),
):

    filepath = upload_service.save_ecg_file(
        file
    )

    return {
        "success": True,
        "file_type": "ECG",
        "file_path": filepath,
    }


@router.post(
    "/echo",
    summary="Upload Echocardiography file",
    description="Uploads an echocardiography video for heart disease prediction.",
    response_description="Echo uploaded successfully.",
)
def upload_echo(
    file: UploadFile = File(...),
    _actor=Depends(get_current_actor),
):

    filepath = upload_service.save_echo_file(
        file
    )

    return {
        "success": True,
        "file_type": "Echo",
        "file_path": filepath,
    }
