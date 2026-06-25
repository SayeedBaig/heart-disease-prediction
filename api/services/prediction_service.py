from fusion_module.pipeline.system_pipeline import SystemPipeline


class PredictionService:
    def __init__(self):
        self.pipeline = SystemPipeline()

    def predict(
        self,
        clinical_data: dict,
        ecg_input,
        echo_input=None,
    ):
        """
        Run complete CardioAI prediction pipeline.
        """

        return self.pipeline.run(
            echo_input=echo_input,
            ecg_input=ecg_input,
            clinical_input=clinical_data,
        )