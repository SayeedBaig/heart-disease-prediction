from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from ecg_module.models.ecg_model_loader import ECGModelLoader
from ecg_module.utils.preprocessing import coerce_ecg_array, prepare_ecg_tensor


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "ecg_attention_calibrated.pth"
)
DEFAULT_NUM_LEADS = 12
DEFAULT_TARGET_LENGTH = 1000
DEFAULT_SAMPLING_RATE = 100

PredictionResponse = Dict[str, Any]
InternalPrediction = Dict[str, Any]

CLASS_LEVEL_MAP = {
    "NORM": "Low",
    "MI": "High",
    "STTC": "Medium",
    "CD": "Medium",
    "HYP": "Medium",
}

CLASS_REASON_MAP = {
    "NORM": (
        "The ECG pattern is closest to a normal rhythm profile, so it is treated as low concern."
    ),
    "MI": (
        "The ECG pattern is most consistent with myocardial infarction related changes, so it is treated as high concern."
    ),
    "STTC": (
        "The ECG pattern shows ST/T change features, so it is treated as medium concern."
    ),
    "CD": (
        "The ECG pattern is most consistent with a conduction disturbance pattern, so it is treated as medium concern."
    ),
    "HYP": (
        "The ECG pattern suggests hypertrophy related changes, so it is treated as medium concern."
    ),
}


def _gaussian_pulse(
    time_axis: np.ndarray,
    center: float,
    width: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((time_axis - center) / width) ** 2)


def simulate_ecg_input(
    num_leads: int = DEFAULT_NUM_LEADS,
    target_length: int = DEFAULT_TARGET_LENGTH,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a realistic synthetic ECG segment for testing."""

    rng = np.random.default_rng(seed)
    duration_seconds = target_length / float(sampling_rate)
    time_axis = np.linspace(
        0.0,
        duration_seconds,
        target_length,
        endpoint=False,
        dtype=np.float32,
    )

    heart_rate_bpm = rng.uniform(58.0, 86.0)
    rr_interval = 60.0 / heart_rate_bpm
    beat_times = []
    current_time = 0.35
    while current_time < duration_seconds + 0.4:
        beat_times.append(current_time)
        current_time += rr_interval + rng.normal(0.0, 0.04)

    lead_scales = rng.normal(1.0, 0.08, size=num_leads)
    lead_shifts = rng.normal(0.0, 0.004, size=num_leads)
    lead_offsets = rng.normal(0.0, 0.015, size=num_leads)

    signal = np.zeros((num_leads, target_length), dtype=np.float32)
    for lead_index in range(num_leads):
        lead_signal = np.zeros(target_length, dtype=np.float32)
        amplitude_scale = float(lead_scales[lead_index])
        shift = float(lead_shifts[lead_index])

        for beat_time in beat_times:
            center = beat_time + shift
            lead_signal += _gaussian_pulse(
                time_axis,
                center - 0.20,
                0.025,
                0.10 * amplitude_scale,
            )
            lead_signal += _gaussian_pulse(
                time_axis,
                center - 0.04,
                0.010,
                -0.14 * amplitude_scale,
            )
            lead_signal += _gaussian_pulse(
                time_axis,
                center,
                0.012,
                1.00 * amplitude_scale,
            )
            lead_signal += _gaussian_pulse(
                time_axis,
                center + 0.04,
                0.015,
                -0.24 * amplitude_scale,
            )
            lead_signal += _gaussian_pulse(
                time_axis,
                center + 0.28,
                0.050,
                0.32 * amplitude_scale,
            )

        baseline_frequency = rng.uniform(0.15, 0.35)
        baseline_phase = rng.uniform(0.0, 2.0 * np.pi)
        baseline = 0.04 * np.sin(
            2.0 * np.pi * baseline_frequency * time_axis + baseline_phase
        )
        noise = rng.normal(0.0, 0.02, size=target_length).astype(np.float32)
        signal[lead_index] = (
            lead_signal + baseline + noise + lead_offsets[lead_index]
        )

    return signal.astype(np.float32)


class ECGAgent:
    """Callable ECG inference agent for offline and real-time workflows."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        label_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.loader = ECGModelLoader(
            model_path=model_path or DEFAULT_MODEL_PATH,
            device=self.device,
        )
        self.artifact = self.loader.load()
        self.model = self.artifact.model
        self.model.eval()

        self.num_leads = self.artifact.num_leads
        self.target_length = self.artifact.target_length
        self.normalization_mean = self.artifact.normalization_mean
        self.normalization_std = self.artifact.normalization_std
        self.label_map = label_map or self.artifact.label_map

        self._stream_lock = Lock()
        self._stream_buffer = np.empty((self.num_leads, 0), dtype=np.float32)

    def __call__(self, ecg_signal: Any) -> PredictionResponse:
        """Allow the agent instance to be called like a function."""

        return self.predict(ecg_signal)

    def predict(self, ecg_signal: Any) -> PredictionResponse:
        """Run single-window ECG inference and return a clean response."""

        try:
            tensor = self._prepare_input_tensor(ecg_signal)
            internal_prediction = self._run_inference(tensor)
            return self._standardize_success_response(internal_prediction)
        except Exception as exc:  # pragma: no cover - API safety
            return self._error_response(exc)

    def predict_simulated(
        self,
        seed: Optional[int] = None,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
    ) -> PredictionResponse:
        """Generate a simulated ECG segment and run inference on it."""

        simulated_signal = simulate_ecg_input(
            num_leads=self.num_leads,
            target_length=self.target_length,
            sampling_rate=sampling_rate,
            seed=seed,
        )
        return self.predict(simulated_signal)

    def predict_realtime(self, ecg_stream_chunk: Any) -> PredictionResponse:
        """Accept one ECG chunk, buffer it, and predict when ready."""

        try:
            chunk = coerce_ecg_array(
                ecg_input=ecg_stream_chunk,
                num_leads=self.num_leads,
                target_length=None,
            )
            buffered_samples = self._append_stream_chunk(chunk)

            if buffered_samples < self.target_length:
                return self._buffering_response(buffered_samples)

            with self._stream_lock:
                window = self._stream_buffer.copy()

            return self.predict(window)
        except Exception as exc:  # pragma: no cover - API safety
            return self._error_response(exc)

    def reset_realtime_buffer(self) -> None:
        """Reset buffered real-time samples."""

        with self._stream_lock:
            self._stream_buffer = np.empty((self.num_leads, 0), dtype=np.float32)

    def _prepare_input_tensor(self, ecg_signal: Any) -> torch.Tensor:
        return prepare_ecg_tensor(
            ecg_input=ecg_signal,
            num_leads=self.num_leads,
            target_length=self.target_length,
            device=self.device,
            mean=self.normalization_mean,
            std=self.normalization_std,
        )

    def _run_inference(self, tensor: torch.Tensor) -> InternalPrediction:
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence_tensor, prediction_tensor = torch.max(probabilities, dim=1)

        prediction = int(prediction_tensor.item())
        confidence = round(float(confidence_tensor.item()), 6)
        label = self.label_map.get(prediction, "UNKNOWN")

        return {
            "prediction": prediction,
            "label": label,
            "confidence": confidence,
        }

    def _append_stream_chunk(self, chunk: np.ndarray) -> int:
        with self._stream_lock:
            self._stream_buffer = np.concatenate((self._stream_buffer, chunk), axis=1)
            if self._stream_buffer.shape[1] > self.target_length:
                self._stream_buffer = self._stream_buffer[:, -self.target_length :]
            return int(self._stream_buffer.shape[1])

    def _standardize_success_response(
        self,
        internal_prediction: InternalPrediction,
    ) -> PredictionResponse:
        label = str(internal_prediction["label"])
        level = CLASS_LEVEL_MAP.get(label, "Medium")
        reason = CLASS_REASON_MAP.get(
            label,
            "The ECG pattern matches an intermediate abnormality profile, so it is treated as medium concern.",
        )

        return {
            "level": level,
            "score": float(internal_prediction["confidence"]),
            "reason": reason,
        }

    def _buffering_response(self, buffered_samples: int) -> PredictionResponse:
        return {
            "level": None,
            "score": None,
            "reason": "The agent is waiting for more ECG samples before it can score the signal.",
            "received_samples": buffered_samples,
            "required_samples": self.target_length,
        }

    @staticmethod
    def _error_response(exc: Exception) -> PredictionResponse:
        return {
            "level": None,
            "score": None,
            "reason": str(exc),
        }


@lru_cache(maxsize=4)
def get_ecg_agent(model_path: Optional[str] = None) -> ECGAgent:
    """Return a cached ECG agent instance for repeated inference calls."""

    resolved_path = str(Path(model_path).resolve()) if model_path else str(DEFAULT_MODEL_PATH)
    return ECGAgent(model_path=resolved_path)


def predict_ecg_signal(
    ecg_signal: Any,
    model_path: Optional[str] = None,
) -> PredictionResponse:
    """Callable helper function for one-shot ECG prediction."""

    agent = get_ecg_agent(model_path=model_path)
    return agent.predict(ecg_signal)
