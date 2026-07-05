from fastapi import APIRouter, File, UploadFile

from api.services.upload_service import UploadService

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
):

    filepath = upload_service.save_echo_file(
        file
    )

    return {
        "success": True,
        "file_type": "Echo",
        "file_path": filepath,
    }