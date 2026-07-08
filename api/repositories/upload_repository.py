from sqlalchemy.orm import Session

from api.models.upload import Upload


class UploadRepository:

    def __init__(self, db: Session):
        self.db = db

    def create_upload(self, upload: Upload) -> Upload:
        self.db.add(upload)
        self.db.commit()
        self.db.refresh(upload)
        return upload

    def get_by_id(self, upload_id: int):
        return (
            self.db.query(Upload)
            .filter(Upload.id == upload_id)
            .first()
        )

    def get_by_patient(self, patient_id: int):
        return (
            self.db.query(Upload)
            .filter(Upload.patient_id == patient_id)
            .all()
        )

    def mark_as_used(self, upload: Upload):
        upload.status = "USED"
        self.db.commit()
        self.db.refresh(upload)
        return upload

    def mark_as_deleted(self, upload: Upload):
        upload.status = "DELETED"
        self.db.commit()
        self.db.refresh(upload)
        return upload