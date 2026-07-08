from pathlib import Path
from datetime import datetime, timedelta


class FileManager:

    @staticmethod
    def get_expired_files(
        directory: Path,
        retention_days: int,
    ):
        expired = []

        cutoff = datetime.now() - timedelta(days=retention_days)

        for file in directory.iterdir():

            if not file.is_file():
                continue

            modified = datetime.fromtimestamp(
                file.stat().st_mtime
            )

            if modified < cutoff:
                expired.append(file)

        return expired

    @staticmethod
    def delete_file(file: Path):

        if file.exists():
            file.unlink()