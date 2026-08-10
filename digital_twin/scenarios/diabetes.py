from copy import deepcopy


def simulate(patient):

    updated = deepcopy(patient)

    updated.glucose *= 0.85

    return updated