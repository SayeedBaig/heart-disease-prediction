import numpy as np

from ecg_module.agent.ecg_agent import ECGAgent


def test_saved_ecg_checkpoint_runs_synthetic_cpu_inference():
    agent = ECGAgent()
    result = agent.predict(np.zeros((12, 1000), dtype=np.float32))

    assert result["level"] in {"Low", "Medium", "High"}
    assert 0 <= result["score"] <= 1
    assert "Class scores:" in result["reason"]
