from __future__ import annotations

import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


DEFAULT_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DEFAULT_MODEL_FILENAME = "ecg_attention_calibrated.pth"
DEFAULT_NUM_LEADS = 12
DEFAULT_TARGET_LENGTH = 1000


class SEBlock(nn.Module):
    """Squeeze-and-excitation block used by the ECG backbone."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class ResConvBlock(nn.Module):
    """Residual convolutional block for 1D ECG feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout: float = 0.3,
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
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.se(self.conv(x)) + residual
        out = self.activation(out)
        out = self.dropout(self.pool(out))
        return out


class TemporalAttentionPool(nn.Module):
    """Attention-based temporal pooling layer."""

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
    """ECG classification backbone recovered from the training notebook."""

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
    """Post-hoc temperature scaling wrapper for calibrated inference."""

    def __init__(self, model: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature


@dataclass(frozen=True)
class ECGModelArtifact:
    """Loaded model bundle used by the agent."""

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    classes: List[str]
    class_to_idx: Dict[str, int]
    label_map: Dict[int, str]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    num_leads: int
    target_length: int
    temperature: float


class ECGModelLoader:
    """Reusable loader for ECG checkpoints stored as `.pth` files."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        num_leads: int = DEFAULT_NUM_LEADS,
        target_length: int = DEFAULT_TARGET_LENGTH,
        default_classes: Optional[List[str]] = None,
        safe_mode: bool = True,
    ) -> None:
        default_path = Path(__file__).resolve().parent / DEFAULT_MODEL_FILENAME
        self.model_path = Path(model_path) if model_path else default_path
        self.device = device or torch.device("cpu")
        self.num_leads = int(num_leads)
        self.target_length = int(target_length)
        self.default_classes = default_classes or DEFAULT_CLASSES.copy()
        self.safe_mode = safe_mode

    def load(self) -> ECGModelArtifact:
        """Load a checkpoint and return a fully initialized inference bundle."""

        checkpoint = self._load_checkpoint()
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            return self._load_from_checkpoint_bundle(checkpoint)
        if isinstance(checkpoint, dict):
            return self._load_from_state_dict(checkpoint)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint.to(self.device)
            model.eval()
            classes = self.default_classes.copy()
            label_map = {index: label for index, label in enumerate(classes)}
            class_to_idx = {label: index for index, label in enumerate(classes)}
            return ECGModelArtifact(
                model=model,
                device=self.device,
                checkpoint_path=self.model_path,
                classes=classes,
                class_to_idx=class_to_idx,
                label_map=label_map,
                normalization_mean=np.zeros((1, self.num_leads, 1), dtype=np.float32),
                normalization_std=np.ones((1, self.num_leads, 1), dtype=np.float32),
                num_leads=self.num_leads,
                target_length=self.target_length,
                temperature=1.0,
            )
        raise TypeError(
            "Unsupported checkpoint format. Expected a checkpoint dict, state_dict, or nn.Module."
        )

    def _load_from_checkpoint_bundle(self, checkpoint: Dict[str, Any]) -> ECGModelArtifact:
        required_keys = {
            "model_state",
            "temperature",
            "train_mean",
            "train_std",
            "classes",
            "class_to_idx",
        }
        missing_keys = sorted(required_keys.difference(checkpoint))
        if missing_keys:
            raise KeyError(
                f"Checkpoint is missing required keys: {', '.join(missing_keys)}."
            )

        classes = [str(label) for label in checkpoint["classes"]]
        class_to_idx = {
            str(label): int(index) for label, index in checkpoint["class_to_idx"].items()
        }
        label_map = {int(index): label for label, index in class_to_idx.items()}
        model = self._build_model(num_classes=len(classes))
        model.load_state_dict(checkpoint["model_state"], strict=True)

        temperature = float(checkpoint["temperature"])
        calibrated_model = TemperatureScaling(model, temperature=1.0)
        calibrated_model.to(self.device)
        with torch.no_grad():
            calibrated_model.temperature.copy_(
                torch.tensor([temperature], dtype=torch.float32, device=self.device)
            )
        calibrated_model.eval()

        train_mean = self._as_numpy_array(checkpoint["train_mean"], "train_mean")
        train_std = self._as_numpy_array(checkpoint["train_std"], "train_std")

        return ECGModelArtifact(
            model=calibrated_model,
            device=self.device,
            checkpoint_path=self.model_path,
            classes=classes,
            class_to_idx=class_to_idx,
            label_map=label_map,
            normalization_mean=train_mean,
            normalization_std=train_std,
            num_leads=self.num_leads,
            target_length=self.target_length,
            temperature=temperature,
        )

    def _load_from_state_dict(self, state_dict: Dict[str, Any]) -> ECGModelArtifact:
        classes = self.default_classes.copy()
        label_map = {index: label for index, label in enumerate(classes)}
        class_to_idx = {label: index for index, label in enumerate(classes)}

        model = self._build_model(num_classes=len(classes))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return ECGModelArtifact(
            model=model,
            device=self.device,
            checkpoint_path=self.model_path,
            classes=classes,
            class_to_idx=class_to_idx,
            label_map=label_map,
            normalization_mean=np.zeros((1, self.num_leads, 1), dtype=np.float32),
            normalization_std=np.ones((1, self.num_leads, 1), dtype=np.float32),
            num_leads=self.num_leads,
            target_length=self.target_length,
            temperature=1.0,
        )

    def _build_model(self, num_classes: int) -> nn.Module:
        model = ECGCNNAttention(
            num_leads=self.num_leads,
            num_classes=num_classes,
            dropout=0.3,
        )
        model.to(self.device)
        model.eval()
        return model

    def _load_checkpoint(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        load_signature = inspect.signature(torch.load)
        supports_weights_only = "weights_only" in load_signature.parameters
        safe_globals_context = nullcontext()
        load_kwargs: Dict[str, Any] = {"map_location": self.device}

        if self.safe_mode:
            safe_globals_factory = getattr(torch.serialization, "safe_globals", None)
            if safe_globals_factory is None or not supports_weights_only:
                raise RuntimeError(
                    "Safe checkpoint loading requires a PyTorch version that supports "
                    "`torch.serialization.safe_globals` and `torch.load(..., weights_only=True)`."
                )
            safe_globals_context = safe_globals_factory(self._get_safe_numpy_globals())
            load_kwargs["weights_only"] = True
        elif supports_weights_only:
            load_kwargs["weights_only"] = False

        with safe_globals_context:
            return torch.load(self.model_path, **load_kwargs)

    @staticmethod
    def _as_numpy_array(value: Any, field_name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive conversion
            raise TypeError(f"Failed to convert `{field_name}` to numpy array.") from exc

        if array.ndim not in (2, 3):
            raise ValueError(
                f"`{field_name}` must be a 2D or 3D array, received shape {array.shape}."
            )
        return array

    @staticmethod
    def _get_safe_numpy_globals() -> List[Any]:
        numpy_globals: List[Any] = [
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
        ]

        dtype_types = {
            type(np.dtype(np.float16)),
            type(np.dtype(np.float32)),
            type(np.dtype(np.float64)),
            type(np.dtype(np.int16)),
            type(np.dtype(np.int32)),
            type(np.dtype(np.int64)),
        }
        numpy_globals.extend(sorted(dtype_types, key=lambda item: item.__name__))
        return numpy_globals
