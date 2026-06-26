from fusion_module.pipeline.system_pipeline import SystemPipeline
from api.utils.logger import get_logger
from api.services.patient_service import PatientService
from api.services.shared_memory_service import SharedMemoryService
from api.utils.validators import validate_patient_data


class PredictionService:
    def __init__(self):
        self.pipeline = SystemPipeline()
        self.patient_service = PatientService()
        self.shared_memory = SharedMemoryService()
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

        errors = validate_patient_data(patient)

        if errors:
            return {
                "success": False,
                "errors": errors,
            }

        result = self.pipeline.run(
            echo_input=echo_input,
            ecg_input=ecg_input,
            clinical_input=patient,
        )
        self.shared_memory.set(
            "latest_prediction",
            result
        )

        self.logger.info("Prediction completed successfully")

        return result