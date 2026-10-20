from clinical_module.agent import clinical_agent


SYNTHETIC_PATIENT = {
    "age": 50,
    "gender": 2,
    "height": 170,
    "weight": 70,
    "ap_hi": 130,
    "ap_lo": 85,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "cholesterol": 2,
    "gluc": 1,
}


def test_clinical_prediction_uses_the_saved_ml_model():
    result, source = clinical_agent.clinical_agent_with_status(SYNTHETIC_PATIENT)

    assert source == "ml"
    assert clinical_agent._stack_model is not None
    assert clinical_agent._scaler is not None
    assert result["level"] in {"Low", "Medium", "High"}
    assert 0 <= result["score"] <= 1
    assert "rule-based fallback" not in result["reason"].lower()


def test_rule_based_fallback_is_reported_internally(monkeypatch):
    monkeypatch.setattr(clinical_agent, "_stack_model", None)

    result, source = clinical_agent.clinical_agent_with_status(SYNTHETIC_PATIENT)

    assert source == "rule_based"
    assert result["reason"].startswith("[Rule-based fallback")
