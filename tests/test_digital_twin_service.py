from api.services.digital_twin_service import DigitalTwinService


def test_prediction_digital_twin_service_runs_all_synthetic_scenarios():
    result = DigitalTwinService().simulate_future(
        {
            "patient_id": "synthetic-patient",
            "age": 50,
            "gender": 2,
            "height": 170,
            "weight": 70,
            "ap_hi": 130,
            "ap_lo": 85,
            "cholesterol": 2,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
        },
        {"fusion": {"risk_percentage": 42.0}},
    )

    assert result["status"] == "success"
    assert 0 <= result["baseline_risk"] <= 1
    assert len(result["simulations"]) == 8
    assert all(0 <= item["risk"] <= 1 for item in result["simulations"])
    assert all({"scenario", "risk", "change"} <= item.keys() for item in result["simulations"])
