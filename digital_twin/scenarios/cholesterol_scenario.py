from copy import deepcopy

from digital_twin.models.patient_profile import PatientProfile


def simulate(patient: PatientProfile):

    updated = deepcopy(patient)

    updated.cholesterol = max(
        0,
        patient.cholesterol - 30
    )

    return updated
