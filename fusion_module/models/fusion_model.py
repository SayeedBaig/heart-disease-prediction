class FusionModel:
    def level_to_num(self, level):
        mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }
        return mapping.get(level, None)

    def num_to_level(self, score):
        if score < 1.5:
            return "Low"
        if score < 2.5:
            return "Medium"
        return "High"

    def predict(self, echo, ecg, clinical):
        """Fuse the agent outputs into a final risk score."""

        echo_level = self.level_to_num(echo.get("level"))
        ecg_level = self.level_to_num(ecg.get("level"))
        clinical_level = self.level_to_num(clinical.get("level"))

        weight_echo = 0.3
        weight_ecg = 0.3
        weight_clinical = 0.4

        values = []
        weights = []

        if echo_level is not None:
            values.append(echo_level * weight_echo * echo.get("score", 0))
            weights.append(weight_echo)

        if ecg_level is not None:
            values.append(ecg_level * weight_ecg * ecg.get("score", 0))
            weights.append(weight_ecg)

        if clinical_level is not None:
            values.append(clinical_level * weight_clinical * clinical.get("score", 0))
            weights.append(weight_clinical)

        if not values:
            return {
                "final_level": "Unknown",
                "risk_percentage": 0.0,
            }

        final_score = sum(values) / sum(weights)
        final_level = self.num_to_level(final_score)
        risk_percentage = final_score / 3 * 100

        return {
            "final_level": final_level,
            "risk_percentage": round(risk_percentage, 2),
        }
