class ModuleOutput:
    """
    Standard output from each module
    """

    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = self.validate_confidence(confidence)

    def validate_confidence(self, confidence):
        if confidence < 0:
            return 0.0
        elif confidence > 1:
            return 1.0
        return confidence

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": self.confidence
        }


class FusionInput:
    """
    Combined input for fusion module
    """

    def __init__(self, echo_output, ecg_output, clinical_output):
        self.echo = echo_output
        self.ecg = ecg_output
        self.clinical = clinical_output

    def to_dict(self):
        return {
            "echo": self.echo.to_dict(),
            "ecg": self.ecg.to_dict(),
            "clinical": self.clinical.to_dict()
        }


# example usage
if __name__ == "__main__":
    echo = ModuleOutput("Abnormal", 0.8)
    ecg = ModuleOutput("Arrhythmia", 0.7)
    clinical = ModuleOutput("High Risk", 0.9)

    fusion_input = FusionInput(echo, ecg, clinical)

    print(fusion_input.to_dict())