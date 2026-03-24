class FusionOutput:
    """
    Final output from fusion module
    """

    def __init__(self, risk_level: str, risk_score: float):
        self.risk_level = risk_level
        self.risk_score = self.validate_score(risk_score)

    def validate_score(self, score):
        if score < 0:
            return 0.0
        elif score > 1:
            return 1.0
        return score

    def to_dict(self):
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score
        }

# example usage
if __name__ == "__main__":
    output = FusionOutput("High", 0.87)
    print(output.to_dict())

#Future Upgrade
# {
#     "risk_level": "High",
#     "risk_score": 0.87,
#     "contributions": {
#         "echo": 0.3,
#         "ecg": 0.3,
#         "clinical": 0.4
#     }
# }