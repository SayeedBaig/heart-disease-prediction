from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch


LOGGER = logging.getLogger(__name__)

EXPECTED_NUM_LEADS = 12
EXPECTED_SIGNAL_LENGTH = 1000


class ECGPreprocessingError(ValueError):
    """Raised when ECG data cannot be prepared for inference."""


def load_ecg_csv(
    file_path: Union[str, Path],
    expected_rows: int = EXPECTED_SIGNAL_LENGTH,
    expected_leads: int = EXPECTED_NUM_LEADS,
) -> np.ndarray:
    """
    Load an ECG CSV file and return a `(12, 1000)` float32 signal array.

    The expected CSV shape is `(1000, 12)` where rows are time samples and
    columns are ECG leads.
    """

    csv_path = Path(file_path)
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"ECG CSV file not found: {csv_path}")

    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as exc:
        raise ECGPreprocessingError(
            f"Failed to read ECG CSV file `{csv_path}`: {exc}"
        ) from exc

    if dataframe.empty:
        raise ECGPreprocessingError("ECG CSV file is empty.")

    if dataframe.isnull().values.any():
        raise ECGPreprocessingError("ECG CSV file contains missing values.")

    try:
        dataframe = dataframe.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ECGPreprocessingError(
            "ECG CSV file contains non-numeric values."
        ) from exc

    if dataframe.shape != (expected_rows, expected_leads):
        raise ECGPreprocessingError(
            f"Invalid ECG CSV shape. Expected ({expected_rows}, {expected_leads}) "
            f"but received {dataframe.shape}."
        )

    ecg_signal = dataframe.to_numpy(dtype=np.float32).T
    LOGGER.debug(
        "Loaded ECG CSV %s with raw shape %s and signal shape %s",
        csv_path,
        dataframe.shape,
        ecg_signal.shape,
    )
    return np.ascontiguousarray(ecg_signal, dtype=np.float32)


def coerce_ecg_signal(
    ecg_signal: Any,
    expected_leads: int = EXPECTED_NUM_LEADS,
    expected_length: int = EXPECTED_SIGNAL_LENGTH,
) -> np.ndarray:
    """
    Coerce in-memory ECG data into a validated `(12, 1000)` float32 array.

    Accepts either `(12, 1000)` or `(1000, 12)` input.
    """

    try:
        signal = np.asarray(ecg_signal, dtype=np.float32)
    except Exception as exc:
        raise ECGPreprocessingError(
            "ECG signal could not be converted into a numeric array."
        ) from exc

    if signal.ndim != 2:
        raise ECGPreprocessingError(
            f"ECG signal must be 2D. Received shape {signal.shape}."
        )

    if signal.shape == (expected_length, expected_leads):
        signal = signal.T
    elif signal.shape != (expected_leads, expected_length):
        raise ECGPreprocessingError(
            f"ECG signal must have shape ({expected_leads}, {expected_length}) "
            f"or ({expected_length}, {expected_leads}). Received {signal.shape}."
        )

    validate_ecg_signal(signal, expected_leads=expected_leads, expected_length=expected_length)
    return np.ascontiguousarray(signal, dtype=np.float32)


def validate_ecg_signal(
    ecg_signal: np.ndarray,
    expected_leads: int = EXPECTED_NUM_LEADS,
    expected_length: int = EXPECTED_SIGNAL_LENGTH,
) -> None:
    """Validate the shape and numerical contents of an ECG signal array."""

    if not isinstance(ecg_signal, np.ndarray):
        raise ECGPreprocessingError("ECG signal must be a numpy array.")

    if ecg_signal.shape != (expected_leads, expected_length):
        raise ECGPreprocessingError(
            f"ECG signal must have shape ({expected_leads}, {expected_length}). "
            f"Received {ecg_signal.shape}."
        )

    if not np.isfinite(ecg_signal).all():
        raise ECGPreprocessingError(
            "ECG signal contains NaN or infinite values."
        )


def normalize_ecg_signal(
    ecg_signal: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization using training statistics."""

    signal = coerce_ecg_signal(ecg_signal)
    mean_array = _prepare_stat_array(train_mean, "train_mean")
    std_array = _prepare_stat_array(train_std, "train_std")
    safe_std = np.where(np.abs(std_array) < 1e-8, 1.0, std_array)
    normalized = (signal - mean_array) / safe_std
    return np.ascontiguousarray(normalized, dtype=np.float32)


def prepare_signal_tensor(
    ecg_signal: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert an ECG signal into a normalized tensor of shape `(1, 12, 1000)`.
    """

    normalized_signal = normalize_ecg_signal(
        ecg_signal=ecg_signal,
        train_mean=train_mean,
        train_std=train_std,
    )
    tensor = torch.from_numpy(normalized_signal).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def _prepare_stat_array(value: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)

    if array.ndim == 3 and array.shape == (1, EXPECTED_NUM_LEADS, 1):
        array = array.squeeze(0)
    elif array.ndim == 2 and array.shape == (EXPECTED_NUM_LEADS, 1):
        pass
    elif array.ndim == 1 and array.shape[0] == EXPECTED_NUM_LEADS:
        array = array.reshape(EXPECTED_NUM_LEADS, 1)
    else:
        raise ECGPreprocessingError(
            f"`{field_name}` must have shape (1, {EXPECTED_NUM_LEADS}, 1), "
            f"({EXPECTED_NUM_LEADS}, 1), or ({EXPECTED_NUM_LEADS},). "
            f"Received {array.shape}."
        )

    return array.astype(np.float32, copy=False)
