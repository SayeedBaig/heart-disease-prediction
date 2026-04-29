from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


class ECGPreprocessingError(ValueError):
    """Raised when ECG input cannot be prepared for inference."""


def coerce_ecg_array(
    ecg_input: Any,
    num_leads: int,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    Convert raw ECG input into a 2D float32 array shaped as `(num_leads, length)`.
    """

    if isinstance(ecg_input, np.ndarray):
        signal = ecg_input.astype(np.float32, copy=False)
    elif isinstance(ecg_input, (list, tuple)):
        try:
            signal = np.asarray(ecg_input, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ECGPreprocessingError(
                "ECG input contains non-numeric values and cannot be converted."
            ) from exc
    else:
        raise ECGPreprocessingError(
            f"Unsupported ECG input type: {type(ecg_input).__name__}."
        )

    if signal.size == 0:
        raise ECGPreprocessingError("ECG input is empty.")
    if signal.ndim not in (1, 2):
        raise ECGPreprocessingError(
            f"ECG input must be 1D or 2D, received {signal.ndim}D."
        )
    if not np.isfinite(signal).all():
        raise ECGPreprocessingError("ECG input contains NaN or infinite values.")

    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    if signal.shape[0] == num_leads:
        aligned = signal
    elif signal.shape[1] == num_leads:
        aligned = signal.T
    else:
        raise ECGPreprocessingError(
            f"ECG input must contain {num_leads} leads. Received shape {signal.shape}."
        )

    if target_length is not None:
        aligned = align_signal_length(aligned, target_length)

    return np.ascontiguousarray(aligned, dtype=np.float32)


def align_signal_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Pad short signals or keep the latest window for long signals."""

    if target_length <= 0:
        raise ECGPreprocessingError("Target length must be greater than zero.")

    current_length = signal.shape[1]
    if current_length == target_length:
        return signal
    if current_length > target_length:
        return signal[:, -target_length:]

    pad_width = target_length - current_length
    return np.pad(signal, ((0, 0), (0, pad_width)), mode="edge")


def normalize_ecg_signal(
    signal: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize an ECG signal using provided statistics or per-lead z-score.
    """

    if mean is None or std is None:
        mean_array = signal.mean(axis=1, keepdims=True)
        std_array = signal.std(axis=1, keepdims=True)
    else:
        mean_array = _prepare_statistic_array(mean, signal.shape[0], "mean")
        std_array = _prepare_statistic_array(std, signal.shape[0], "std")

    safe_std = np.where(np.abs(std_array) < eps, 1.0, std_array)
    normalized = (signal - mean_array) / safe_std
    return normalized.astype(np.float32, copy=False)


def prepare_ecg_tensor(
    ecg_input: Any,
    num_leads: int,
    target_length: int,
    device: torch.device,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Convert ECG input into a normalized tensor shaped `(1, num_leads, target_length)`.
    """

    signal = coerce_ecg_array(
        ecg_input=ecg_input,
        num_leads=num_leads,
        target_length=target_length,
    )
    normalized = normalize_ecg_signal(signal, mean=mean, std=std)
    return torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)


def _prepare_statistic_array(
    value: np.ndarray,
    num_leads: int,
    field_name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ECGPreprocessingError(
            f"Normalization `{field_name}` must be 1D, 2D, or 3D, received shape {array.shape}."
        )

    if array.shape[0] != num_leads:
        raise ECGPreprocessingError(
            f"Normalization `{field_name}` is incompatible with {num_leads} leads: {array.shape}."
        )
    if array.shape[1] not in (1,):
        raise ECGPreprocessingError(
            "Normalization "
            f"`{field_name}` must broadcast across time, received shape {array.shape}."
        )

    return array
