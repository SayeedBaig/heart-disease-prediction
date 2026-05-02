from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)

DEFAULT_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DEFAULT_MODEL_FILENAME = "ecg_attention_calibrated.pth"
DEFAULT_NUM_LEADS = 12
DEFAULT_SIGNAL_LENGTH = 1000


class ModelLoadingError(RuntimeError):
    """Raised when a checkpoint cannot be loaded for inference."""


class SEBlock(nn.Module):
    """Squeeze-and-excitation block used by the ECG backbone."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class ResConvBlock(nn.Module):
    """Residual 1D convolution block for ECG feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.se = SEBlock(out_channels)
        self.proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        features = self.se(self.conv(x)) + residual
        features = self.activation(features)
        features = self.pool(features)
        return self.dropout(features)


class TemporalAttentionPool(nn.Module):
    """Attention-based temporal pooling for ECG sequences."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels // 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=-1)
        return (x * weights).sum(dim=-1)


class ECGCNNAttention(nn.Module):
    """PTB-XL ECG classifier architecture recovered from training."""

    def __init__(
        self,
        num_leads: int = DEFAULT_NUM_LEADS,
        num_classes: int = len(DEFAULT_CLASSES),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResConvBlock(num_leads, 32, kernel_size=11, pool_size=2, dropout=dropout),
            ResConvBlock(32, 64, kernel_size=7, pool_size=2, dropout=dropout),
            ResConvBlock(64, 128, kernel_size=5, pool_size=2, dropout=dropout),
            ResConvBlock(128, 256, kernel_size=3, pool_size=2, dropout=dropout),
            ResConvBlock(256, 256, kernel_size=3, pool_size=2, dropout=dropout),
        )
        self.pool = TemporalAttentionPool(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)


class TemperatureScaling(nn.Module):
    """Apply learned temperature scaling during inference."""

    def __init__(self, model: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(
            torch.tensor([temperature], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature


@dataclass(frozen=True)
class LoadedECGModel:
    """Container for a fully loaded ECG model artifact."""

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    class_labels: List[str]
    label_map: Dict[int, str]
    train_mean: np.ndarray
    train_std: np.ndarray
    num_leads: int
    signal_length: int


class ECGModelLoader:
    """Load a trained ECG checkpoint and restore it for CPU inference."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        num_leads: int = DEFAULT_NUM_LEADS,
        signal_length: int = DEFAULT_SIGNAL_LENGTH,
    ) -> None:
        default_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / DEFAULT_MODEL_FILENAME
        )
        self.model_path = Path(model_path) if model_path else default_path
        self.device = device or torch.device("cpu")
        self.num_leads = num_leads
        self.signal_length = signal_length

    def load(self) -> LoadedECGModel:
        """Load the checkpoint, weights, normalization stats, and labels."""

        LOGGER.info("Loading ECG checkpoint from %s", self.model_path)

        if not self.model_path.exists():
            raise ModelLoadingError(
                f"Model checkpoint not found: {self.model_path}"
            )

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,
            )
        except Exception as exc:
            raise ModelLoadingError(
                f"Failed to load checkpoint from {self.model_path}: {exc}"
            ) from exc

        if not isinstance(checkpoint, dict):
            raise ModelLoadingError(
                "Unsupported checkpoint format. Expected a checkpoint dictionary."
            )

        required_keys = {"model_state", "train_mean", "train_std"}
        missing_keys = sorted(required_keys.difference(checkpoint.keys()))
        if missing_keys:
            raise ModelLoadingError(
                f"Checkpoint is missing required keys: {', '.join(missing_keys)}"
            )

        class_labels = self._resolve_class_labels(checkpoint)
        label_map = self._resolve_label_map(checkpoint, class_labels)
        temperature = float(checkpoint.get("temperature", 1.0))

        model = self._build_model(num_classes=len(class_labels))
        try:
            model.model.load_state_dict(checkpoint["model_state"], strict=True)
        except Exception as exc:
            raise ModelLoadingError(
                f"Failed to load model weights: {exc}"
            ) from exc

        with torch.no_grad():
            model.temperature.fill_(temperature)
        model.to(self.device)
        model.eval()

        train_mean = self._prepare_stat_array(
            checkpoint["train_mean"],
            field_name="train_mean",
        )
        train_std = self._prepare_stat_array(
            checkpoint["train_std"],
            field_name="train_std",
        )

        LOGGER.info(
            "Loaded ECG model with %d classes and temperature %.4f",
            len(class_labels),
            temperature,
        )

        return LoadedECGModel(
            model=model,
            device=self.device,
            checkpoint_path=self.model_path,
            class_labels=class_labels,
            label_map=label_map,
            train_mean=train_mean,
            train_std=train_std,
            num_leads=self.num_leads,
            signal_length=self.signal_length,
        )

    def _build_model(self, num_classes: int) -> TemperatureScaling:
        base_model = ECGCNNAttention(
            num_leads=self.num_leads,
            num_classes=num_classes,
        )
        base_model.to(self.device)
        base_model.eval()
        return TemperatureScaling(base_model, temperature=1.0)

    @staticmethod
    def _resolve_class_labels(checkpoint: Dict[str, Any]) -> List[str]:
        classes = checkpoint.get("classes", DEFAULT_CLASSES)
        if not isinstance(classes, list) or not classes:
            LOGGER.warning(
                "Checkpoint does not contain valid class labels; falling back to defaults."
            )
            return DEFAULT_CLASSES.copy()
        return [str(label) for label in classes]

    @staticmethod
    def _resolve_label_map(
        checkpoint: Dict[str, Any],
        class_labels: List[str],
    ) -> Dict[int, str]:
        class_to_idx = checkpoint.get("class_to_idx")
        if not isinstance(class_to_idx, dict):
            LOGGER.warning(
                "Checkpoint does not contain a valid class_to_idx mapping; rebuilding from classes."
            )
            class_to_idx = {label: index for index, label in enumerate(class_labels)}

        try:
            label_map = {
                int(index): str(label)
                for label, index in class_to_idx.items()
            }
        except Exception as exc:
            raise ModelLoadingError(
                "Checkpoint contains an invalid `class_to_idx` mapping."
            ) from exc

        if len(label_map) != len(class_labels):
            LOGGER.warning(
                "Checkpoint class metadata is inconsistent; rebuilding label_map from class order."
            )
            label_map = {index: label for index, label in enumerate(class_labels)}

        return label_map

    def _prepare_stat_array(self, value: Any, field_name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float32)
        except Exception as exc:
            raise ModelLoadingError(
                f"Failed to convert `{field_name}` into a numpy array."
            ) from exc

        if array.ndim == 3 and array.shape == (1, self.num_leads, 1):
            array = array.squeeze(0)
        elif array.ndim == 2 and array.shape == (self.num_leads, 1):
            pass
        elif array.ndim == 1 and array.shape[0] == self.num_leads:
            array = array.reshape(self.num_leads, 1)
        else:
            raise ModelLoadingError(
                f"`{field_name}` must have shape (1, {self.num_leads}, 1), "
                f"({self.num_leads}, 1), or ({self.num_leads},). "
                f"Received {array.shape}."
            )

        return array.astype(np.float32, copy=False)
