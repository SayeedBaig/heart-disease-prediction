from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

DEFAULT_NUM_LEADS = 12
DEFAULT_SIGNAL_LENGTH = 1000


class ECGImageProcessingError(ValueError):
    """Raised when an ECG image cannot be converted into signal form."""


def extract_signal_from_image(
    image_path: Union[str, Path],
    num_leads: int = DEFAULT_NUM_LEADS,
    target_length: int = DEFAULT_SIGNAL_LENGTH,
) -> np.ndarray:
    """
    Convert an ECG image into an approximate `(12, 1000)` signal array.

    This method is intentionally approximate and intended for real-time or
    convenience mode rather than clinical-grade waveform reconstruction.
    """

    if cv2 is None:
        raise ECGImageProcessingError(
            "OpenCV is required for image-based ECG inference. "
            "Install `opencv-python` or `opencv-python-headless`."
        )

    image_file = Path(image_path)
    if not image_file.exists() or not image_file.is_file():
        raise FileNotFoundError(f"ECG image file not found: {image_file}")

    image = cv2.imread(str(image_file))
    if image is None:
        raise ECGImageProcessingError(
            f"Failed to read ECG image file: {image_file}"
        )

    LOGGER.debug("Loaded ECG image %s with shape %s", image_file, image.shape)
    waveform_mask = _build_waveform_mask(image)
    signals = _extract_lead_signals(
        waveform_mask=waveform_mask,
        num_leads=num_leads,
        target_length=target_length,
    )
    return signals


def _build_waveform_mask(image: np.ndarray) -> np.ndarray:
    """Create a binary mask that highlights the ECG waveform."""

    if cv2 is None:  # pragma: no cover - defensive guard
        raise ECGImageProcessingError("OpenCV is unavailable.")

    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image.copy()

    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    cleaned = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )
    edges = cv2.Canny(cleaned, 50, 150)
    waveform_mask = cv2.bitwise_or(cleaned, edges)
    return waveform_mask


def _extract_lead_signals(
    waveform_mask: np.ndarray,
    num_leads: int,
    target_length: int,
) -> np.ndarray:
    """Split the waveform mask into lead bands and extract one signal per lead."""

    height, _ = waveform_mask.shape
    boundaries = np.linspace(0, height, num_leads + 1, dtype=int)
    lead_signals = []

    for lead_index in range(num_leads):
        top = boundaries[lead_index]
        bottom = boundaries[lead_index + 1]
        region = waveform_mask[top:bottom, :]

        if region.size == 0:
            raise ECGImageProcessingError(
                f"Lead region {lead_index} is empty after image splitting."
            )

        lead_signal = _trace_waveform(region)
        resampled_signal = _resample_signal(lead_signal, target_length)
        lead_signals.append(resampled_signal)

    return np.ascontiguousarray(np.stack(lead_signals, axis=0), dtype=np.float32)


def _trace_waveform(region: np.ndarray) -> np.ndarray:
    """Extract a 1D waveform by tracing pixel positions column by column."""

    height, width = region.shape
    if width < 2:
        raise ECGImageProcessingError(
            "ECG image region is too narrow to extract a waveform."
        )

    center_line = (height - 1) / 2.0
    waveform = np.full(width, np.nan, dtype=np.float32)

    for column_index in range(width):
        active_pixels = np.flatnonzero(region[:, column_index] > 0)
        if active_pixels.size == 0:
            continue

        pixel_position = float(np.median(active_pixels))
        amplitude = (center_line - pixel_position) / max(center_line, 1.0)
        waveform[column_index] = amplitude

    waveform = _fill_missing_samples(waveform)
    waveform = _smooth_signal(waveform)
    waveform -= float(np.mean(waveform))

    scale = float(np.max(np.abs(waveform)))
    if scale > 1e-6:
        waveform /= scale

    return waveform.astype(np.float32, copy=False)


def _fill_missing_samples(signal: np.ndarray) -> np.ndarray:
    """Interpolate columns where no waveform pixel was detected."""

    valid_indices = np.flatnonzero(~np.isnan(signal))
    if valid_indices.size == 0:
        LOGGER.warning(
            "No waveform pixels detected in one lead region; using flat fallback signal."
        )
        return np.zeros_like(signal, dtype=np.float32)

    if valid_indices.size == 1:
        signal[:] = signal[valid_indices[0]]
        return signal.astype(np.float32, copy=False)

    all_indices = np.arange(signal.shape[0])
    signal = np.interp(all_indices, valid_indices, signal[valid_indices])
    return signal.astype(np.float32, copy=False)


def _smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply light smoothing to stabilize the extracted waveform."""

    if window_size <= 1 or signal.shape[0] < window_size:
        return signal.astype(np.float32, copy=False)

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(signal, kernel, mode="same")
    return smoothed.astype(np.float32, copy=False)


def _resample_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Resize a 1D signal to a fixed target length using interpolation."""

    if target_length <= 0:
        raise ECGImageProcessingError("Target signal length must be positive.")

    if signal.shape[0] == target_length:
        return signal.astype(np.float32, copy=False)

    source_axis = np.linspace(0.0, 1.0, num=signal.shape[0], dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = np.interp(target_axis, source_axis, signal)
    return resampled.astype(np.float32, copy=False)
