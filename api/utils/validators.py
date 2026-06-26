from clinical_module.agent.clinical_agent import validate_clinical_input


def validate_patient_data(patient_data: dict) -> list[str]:
    """
    Validate patient clinical information.

    Returns:
        Empty list if valid.
        List of validation errors otherwise.
    """
    return validate_clinical_input(patient_data)