from fusion_module.pipeline.system_pipeline import SystemPipeline
from api.utils.logger import get_logger
from api.services.patient_service import PatientService


class PredictionService:
    def __init__(self):
        self.pipeline = SystemPipeline()
        self.patient_service = PatientService()
        self.logger = get_logger(__name__)

    def predict(
        self,
        clinical_data: dict,
        ecg_input,
        echo_input=None,
    ):
        """
        Run complete CardioAI prediction pipeline.
        """

        self.logger.info("Starting prediction pipeline")

        patient = self.patient_service.prepare_patient(
            clinical_data
        )

        result = self.pipeline.run(
            echo_input=echo_input,
            ecg_input=ecg_input,
            clinical_input=patient,
        )

        self.logger.info("Prediction completed successfully")

        return result