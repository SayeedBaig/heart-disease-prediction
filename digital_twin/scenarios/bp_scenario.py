def lower_bp(patient):

    patient.bp -= 15

    patient.risk *= 0.92

    return patient