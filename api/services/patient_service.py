from typing import Dict, Any


class PatientService:
    """
    Handles patient data preparation before it is
    passed to prediction, RAG, reports or Digital Twin.
    """

    def prepare_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and standardize patient information.
        """

        patient = patient_data.copy()

        return patient