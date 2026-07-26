
from fusion_module.pipeline.system_pipeline import SystemPipeline

from api.repositories.prediction_repository import PredictionRepository
from api.services.digital_twin_service import DigitalTwinService
from api.services.patient_service import PatientService
from api.services.rag_service import RAGService
from api.services.response_builder import ResponseBuilder
from api.services.shared_memory_service import SharedMemoryService

from api.utils.logger import get_logger
from api.utils.validators import validate_patient_data


class PredictionService:
    def __init__(self, db):
        self.pipeline = SystemPipeline()
        self.patient_service = PatientService()
        self.shared_memory = SharedMemoryService()
        self.rag_service = RAGService()
        self.digital_twin_service = DigitalTwinService()
        self.response_builder = ResponseBuilder()
        self.prediction_repository = PredictionRepository(db)
        self.logger = get_logger(__name__)

    def predict(
        self,
        clinical_data: dict,
        ecg_input,
        echo_input=None,
        patient_record=None,
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

        prediction = self.pipeline.run(
            echo_input=echo_input,
            ecg_input=ecg_input,
            clinical_input=patient,
        )

        self.shared_memory.set(
            "latest_prediction",
            prediction,
        )

        explanation = self.rag_service.get_explanation(
            prediction
        )

        self.shared_memory.set(
            "latest_explanation",
            explanation,
        )

        digital_twin = self.digital_twin_service.simulate_future(
            patient,
            prediction
        )

        self.shared_memory.set(
            "latest_digital_twin",
            digital_twin,
        )

        response = self.response_builder.build(
            prediction=prediction,
            explanation=explanation,
            digital_twin=digital_twin,
        )

        self.logger.info("Prediction completed successfully")

        if patient_record:
            prediction_db = self.prediction_repository.save(
                patient_db_id=patient_record.id,
                prediction=response,
            )
            response["prediction_id"] = prediction_db.id

        return response