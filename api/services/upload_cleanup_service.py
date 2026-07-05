from api.config.upload_config import (
    ECG_UPLOAD_DIR,
    ECHO_UPLOAD_DIR,
    FILE_RETENTION_DAYS,
)
from api.utils.file_manager import FileManager
from api.utils.logger import get_logger


class UploadCleanupService:

    def __init__(self):
        self.logger = get_logger(__name__)

    def cleanup_ecg_uploads(self):

        files = FileManager.get_expired_files(
            ECG_UPLOAD_DIR,
            FILE_RETENTION_DAYS,
        )

        for file in files:
            FileManager.delete_file(file)

        self.logger.info(
            f"ECG cleanup completed. Deleted {len(files)} file(s)."
        )

        return len(files)

    def cleanup_echo_uploads(self):

        files = FileManager.get_expired_files(
            ECHO_UPLOAD_DIR,
            FILE_RETENTION_DAYS,
        )

        for file in files:
            FileManager.delete_file(file)

        self.logger.info(
            f"Echo cleanup completed. Deleted {len(files)} file(s)."
        )

        return len(files)

    def cleanup_all(self):

        ecg_deleted = self.cleanup_ecg_uploads()
        echo_deleted = self.cleanup_echo_uploads()

        self.logger.info(
            f"Upload Cleanup Summary | ECG: {ecg_deleted} | Echo: {echo_deleted}"
        )

        return {
            "ecg_deleted": ecg_deleted,
            "echo_deleted": echo_deleted,
            "total_deleted": ecg_deleted + echo_deleted,
        }