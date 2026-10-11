
from fusion_module.pipeline.system_pipeline import SystemPipeline

from api.repositories.prediction_repository import PredictionRepository
from api.services.digital_twin_service import DigitalTwinService
from api.services.patient_service import PatientService
from api.services.rag_service import RAGService
from api.services.response_builder import ResponseBuilder
from api.services.shared_memory_service import SharedMemoryService
from clinical_module.alerts.clinical_alert_engine import (
    evaluate_clinical_alerts,
    alerts_to_dicts,
)

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

        # ── Clinical alert evaluation ─────────────────────────────────────
        # Evaluate against the raw patient inputs, not the ML prediction score.
        # A low ML risk score must never suppress a clinical alert.
        try:
            # The raw clinical dict uses ap_hi / ap_lo (mmHg) and categorical
            # cholesterol (1–3) / gluc (1–3) from the ClinicalInput schema.
            raw_systolic = clinical_data.get("ap_hi")
            raw_diastolic = clinical_data.get("ap_lo")
            # cholesterol and gluc are categorical integers (1/2/3).
            # The alert engine accepts both; glucose is categorical here so
            # we pass it to the cholesterol checker, not the mg/dL checker.
            raw_cholesterol_cat = clinical_data.get("cholesterol")
            # gluc categorical (1=normal,2=above normal,3=well above normal)
            # is structurally identical to cholesterol category — evaluate separately.
            raw_gluc_cat = clinical_data.get("gluc")

            alerts = evaluate_clinical_alerts(
                systolic_bp=float(raw_systolic) if raw_systolic is not None else None,
                diastolic_bp=float(raw_diastolic) if raw_diastolic is not None else None,
                glucose=None,  # raw mg/dL not available; categorical gluc handled below
                cholesterol_category=int(raw_cholesterol_cat) if raw_cholesterol_cat is not None else None,
                is_hypothetical=False,
                context_id=str(response.get("prediction_id", "latest")),
            )
            # Evaluate the glucose categorical separately (same encoding as cholesterol)
            if raw_gluc_cat is not None:
                from clinical_module.alerts.clinical_alert_engine import _cholesterol_alert as _cat_alert
                gluc_alert = _cat_alert(
                    raw_gluc_cat,
                    is_hypothetical=False,
                    context_id=str(response.get("prediction_id", "latest")),
                )
                if gluc_alert:
                    # Re-label as glucose
                    gluc_alert.alert_id = gluc_alert.alert_id.replace("CHOLESTEROL", "GLUCOSE")
                    gluc_alert.parameter = "Blood Glucose (categorical)"
                    gluc_alert.category = "Glucose"
                    alerts.append(gluc_alert)

            alert_dicts = alerts_to_dicts(alerts)
        except Exception:
            self.logger.exception("Clinical alert evaluation failed — alerts omitted from response")
            alert_dicts = []

        response["clinical_alerts"] = alert_dicts

        self.shared_memory.set("latest_clinical_alerts", alert_dicts)

        self.logger.info("Prediction completed successfully")

        if patient_record:
            prediction_db = self.prediction_repository.save(
                patient_db_id=patient_record.id,
                prediction=response,
            )
            response["prediction_id"] = prediction_db.id

        return response