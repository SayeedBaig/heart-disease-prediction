def improve_cholesterol(patient):

    patient.cholesterol -= 30

    patient.risk *= 0.90

    return patient