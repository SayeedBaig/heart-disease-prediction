from pathlib import Path
import shutil
import uuid

from fastapi import UploadFile, HTTPException

from api.config.upload_config import (
    ECG_UPLOAD_DIR,
    ECHO_UPLOAD_DIR,
    ALLOWED_ECG_EXTENSIONS,
    ALLOWED_ECHO_EXTENSIONS,
)

from api.utils.logger import get_logger


class UploadService:

    def __init__(self):
        self.logger = get_logger(__name__)

    def save_ecg_file(
        self,
        file: UploadFile,
    ) -> str:

        return self._save_file(
            file=file,
            upload_dir=ECG_UPLOAD_DIR,
            allowed_extensions=ALLOWED_ECG_EXTENSIONS,
        )

    def save_echo_file(
        self,
        file: UploadFile,
    ) -> str:

        return self._save_file(
            file=file,
            upload_dir=ECHO_UPLOAD_DIR,
            allowed_extensions=ALLOWED_ECHO_EXTENSIONS,
        )

    def _save_file(
        self,
        file: UploadFile,
        upload_dir: Path,
        allowed_extensions: set,
    ) -> str:

        extension = Path(file.filename).suffix.lower()

        if extension not in allowed_extensions:

            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {extension}",
            )

        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{uuid.uuid4()}{extension}"
        )

        filepath = upload_dir / filename

        with filepath.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        self.logger.info(
            f"Uploaded file saved: {filepath}"
        )

        return str(filepath)