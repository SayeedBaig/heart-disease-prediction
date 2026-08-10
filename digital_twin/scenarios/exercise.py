from copy import deepcopy


def simulate(patient):

    updated = deepcopy(patient)

    updated.bmi = max(

        0,

        patient.bmi - 2

    )

    updated.weight = max(

        0,

        patient.weight - 5

    )

    updated.exercise_level = "High"

    return updated