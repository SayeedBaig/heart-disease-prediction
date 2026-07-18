class RiskImprovementMeter:
    """
    Calculates overall improvement between
    current and projected risk.
    """

    def calculate(self, current_risk, projected_risk):

        improvement = max(0, current_risk - projected_risk)

        improvement_percentage = 0

        if current_risk > 0:
            improvement_percentage = (
                improvement / current_risk
            ) * 100

        if improvement_percentage >= 30:
            level = "Excellent"

        elif improvement_percentage >= 20:
            level = "Good"

        elif improvement_percentage >= 10:
            level = "Moderate"

        else:
            level = "Minimal"

        return {

            "current_risk": round(current_risk, 2),

            "projected_risk": round(projected_risk, 2),

            "risk_reduction": round(improvement, 2),

            "improvement_percentage": round(
                improvement_percentage,
                2
            ),

            "level": level,

            "color": (
                "green"
                if level == "Excellent"
                else
                "yellow"
                if level == "Good"
                else
                "orange"
                if level == "Moderate"
                else
                "red"
            )

        }