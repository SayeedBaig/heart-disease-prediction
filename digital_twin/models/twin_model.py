class PatientTwin:

    def __init__(self, data):

        self.age = data.get("age", 0)

        self.bp = data.get("bp", 0)

        self.cholesterol = data.get("cholesterol", 0)

        self.bmi = data.get("bmi", 0)

        self.smoking = data.get("smoking", 0)

        self.risk = data.get("risk_score", 0.0)