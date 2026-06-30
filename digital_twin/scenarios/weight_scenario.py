from copy import deepcopy

from digital_twin.models.patient_profile import PatientProfile


def simulate(patient: PatientProfile):

    updated = deepcopy(patient)

    updated.weight = max(
        0,
        patient.weight - 10
    )

    updated.bmi = max(
        0,
        patient.bmi - 3
    )

    return updated
