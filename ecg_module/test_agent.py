from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_module.agent.ecg_agent import ECGAgent, simulate_ecg_input


def run_realtime_demo(agent: ECGAgent, signal) -> dict:
    final_result = {}
    chunk_size = 200

    for start_index in range(0, signal.shape[1], chunk_size):
        chunk = signal[:, start_index : start_index + chunk_size]
        final_result = agent.predict_realtime(chunk)

    return final_result


def main() -> None:
    model_path = Path(__file__).resolve().parent / "models" / "ecg_attention_calibrated.pth"
    agent = ECGAgent(model_path=model_path)

    simulated_signal = simulate_ecg_input(
        num_leads=12,
        target_length=1000,
        sampling_rate=100,
        seed=42,
    )

    prediction_result = agent.predict(simulated_signal)
    agent.reset_realtime_buffer()
    realtime_result = run_realtime_demo(agent, simulated_signal)

    print("Offline Prediction:")
    print(json.dumps(prediction_result, indent=4))
    print()
    print("Real-Time Prediction:")
    print(json.dumps(realtime_result, indent=4))


if __name__ == "__main__":
    main()
