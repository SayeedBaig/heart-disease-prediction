from copy import deepcopy

from digital_twin.models.patient_profile import PatientProfile


def simulate(patient: PatientProfile):

    updated = deepcopy(patient)

    updated.systolic_bp = max(
        0,
        patient.systolic_bp - 15
    )

    updated.diastolic_bp = max(
        0,
        patient.diastolic_bp - 10
    )

<<<<<<< HEAD
    return updated
=======
    return updated
>>>>>>> develop
