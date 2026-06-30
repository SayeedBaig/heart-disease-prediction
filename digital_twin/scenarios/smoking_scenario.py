from copy import deepcopy

from digital_twin.models.patient_profile import PatientProfile


def simulate(patient: PatientProfile):

    updated = deepcopy(patient)

    updated.smoking_status = False

    return updated
