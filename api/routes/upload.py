from fastapi import APIRouter, File, UploadFile

from api.services.upload_service import UploadService

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)

upload_service = UploadService()


@router.post("/ecg")
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


@router.post("/echo")
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