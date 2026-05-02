from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, TypedDict, Union

import numpy as np
import torch

from ecg_module.model.ecg_model_loader import ECGModelLoader, ModelLoadingError
from ecg_module.utils.image_to_signal import (
    ECGImageProcessingError,
    extract_signal_from_image,
)
from ecg_module.utils.preprocessing import (
    ECGPreprocessingError,
    coerce_ecg_signal,
    load_ecg_csv,
    prepare_signal_tensor,
)


LOGGER = logging.getLogger(__name__)

SignalSource = Literal["signal", "csv", "image"]

DISPLAY_LABEL_MAP: Dict[str, str] = {
    "NORM": "Normal rhythm",
    "MI": "Myocardial infarction",
    "AMI": "Acute myocardial infarction",
    "STEMI": "ST-elevation myocardial infarction",
    "STTC": "ST/T changes",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}

CLINICAL_LEVEL_MAP: Dict[str, str] = {
    "NORM": "Low",
    "STTC": "Medium",
    "HYP": "Medium",
    "CD": "Medium",
    "MI": "High",
    "AMI": "High",
    "STEMI": "High",
}

ERROR_LEVEL = "Low"
ERROR_SCORE = 0.0
LOW_CONFIDENCE_THRESHOLD = 0.58
MEDIUM_CONFIDENCE_THRESHOLD = 0.70


class ECGPredictionResponse(TypedDict):
    """Strict response contract for ECG predictions."""

    Level: str
    Score: float
    Reason: str


class ECGAgent:
    """Production-ready ECG agent for CSV and image-based inference."""

    def __init__(self, model_path: Union[str, Path, None] = None) -> None:
        self.device = torch.device("cpu")
        self.loader = ECGModelLoader(model_path=model_path, device=self.device)

        try:
            self.artifact = self.loader.load()
        except ModelLoadingError:
            LOGGER.exception(
                "Failed to initialize ECGAgent from checkpoint %s",
                self.loader.model_path,
            )
            raise

        self.model = self.artifact.model
        self.model.eval()
        self.label_map = self.artifact.label_map
        self.class_labels = self.artifact.class_labels

    def predict_from_csv(self, file_path: Union[str, Path]) -> ECGPredictionResponse:
        """Load ECG data from CSV and run prediction."""

        LOGGER.info("Starting ECG CSV prediction for %s", file_path)
        try:
            ecg_signal = load_ecg_csv(file_path=file_path)
            return self._predict_signal(ecg_signal=ecg_signal, source="csv")
        except (FileNotFoundError, ECGPreprocessingError) as exc:
            LOGGER.warning("CSV prediction failed for %s: %s", file_path, exc)
            return self._build_error_response(f"Prediction failed: {exc}")
        except Exception as exc:
            LOGGER.exception("Unexpected CSV prediction failure for %s", file_path)
            return self._build_error_response(
                f"Prediction failed: unexpected CSV inference error: {exc}"
            )

    def predict_from_image(
        self,
        image_path: Union[str, Path],
    ) -> ECGPredictionResponse:
        """Load ECG data from image, extract signal, and run prediction."""

        LOGGER.info("Starting ECG image prediction for %s", image_path)
        try:
            ecg_signal = extract_signal_from_image(image_path=image_path)
            return self._predict_signal(ecg_signal=ecg_signal, source="image")
        except (FileNotFoundError, ECGImageProcessingError) as exc:
            LOGGER.warning("Image prediction failed for %s: %s", image_path, exc)
            return self._build_error_response(f"Prediction failed: {exc}")
        except Exception as exc:
            LOGGER.exception("Unexpected image prediction failure for %s", image_path)
            return self._build_error_response(
                f"Prediction failed: unexpected image inference error: {exc}"
            )

    def predict(self, ecg_signal: np.ndarray) -> ECGPredictionResponse:
        """Run prediction on an in-memory ECG signal array."""

        LOGGER.info("Starting ECG signal-array prediction.")
        try:
            return self._predict_signal(ecg_signal=ecg_signal, source="signal")
        except ECGPreprocessingError as exc:
            LOGGER.warning("Signal-array prediction failed: %s", exc)
            return self._build_error_response(f"Prediction failed: {exc}")
        except Exception as exc:
            LOGGER.exception("Unexpected signal-array prediction failure.")
            return self._build_error_response(
                f"Prediction failed: unexpected signal inference error: {exc}"
            )

    def _predict_signal(
        self,
        ecg_signal: np.ndarray,
        source: SignalSource,
    ) -> ECGPredictionResponse:
        """Shared inference path for CSV, image, and in-memory signal inputs."""

        signal = coerce_ecg_signal(ecg_signal)
        input_tensor = prepare_signal_tensor(
            ecg_signal=signal,
            train_mean=self.artifact.train_mean,
            train_std=self.artifact.train_std,
            device=self.device,
        )

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence_tensor, prediction_tensor = torch.max(probabilities, dim=1)

        prediction_index = int(prediction_tensor.item())
        confidence = float(confidence_tensor.item())
        predicted_label = self.label_map.get(prediction_index, "UNKNOWN")
        clinical_level = self._map_clinical_level(predicted_label)
        final_level = self._adjust_level_by_confidence(clinical_level, confidence)
        probability_values = probabilities.squeeze(0).cpu().tolist()
        reason = self._build_reason(
            predicted_label=predicted_label,
            confidence=confidence,
            clinical_level=clinical_level,
            final_level=final_level,
            probabilities=probability_values,
            source=source,
        )

        LOGGER.info(
            "Completed ECG %s prediction with label %s, confidence %.4f, level %s",
            source,
            predicted_label,
            confidence,
            final_level,
        )

        return {
            "Level": final_level,
            "Score": round(confidence, 6),
            "Reason": reason,
        }

    @staticmethod
    def _build_error_response(message: str) -> ECGPredictionResponse:
        """Return a strict error-shaped response."""

        return {
            "Level": ERROR_LEVEL,
            "Score": ERROR_SCORE,
            "Reason": message,
        }

    @staticmethod
    def _map_clinical_level(predicted_label: str) -> str:
        """Map the predicted class label to a clinical risk level."""

        return CLINICAL_LEVEL_MAP.get(predicted_label, "Medium")

    @staticmethod
    def _adjust_level_by_confidence(
        mapped_level: str,
        confidence: float,
    ) -> str:
        """Adjust the alert level according to confidence thresholds."""

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            return "Low"
        if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
            return "Medium"
        return mapped_level

    def _build_reason(
        self,
        predicted_label: str,
        confidence: float,
        clinical_level: str,
        final_level: str,
        probabilities: List[float],
        source: SignalSource,
    ) -> str:
        """Generate a clinically readable explanation from prediction outputs."""

        display_name = DISPLAY_LABEL_MAP.get(predicted_label, predicted_label)
        confidence_phrase = self._confidence_phrase(confidence)
        base_reason = self._base_clinical_reason(predicted_label, confidence_phrase)
        level_sentence = self._level_adjustment_reason(
            clinical_level=clinical_level,
            final_level=final_level,
            confidence=confidence,
        )
        score_summary = self._format_score_summary(probabilities)
        source_sentence = ""

        if source == "image":
            source_sentence = (
                " This result comes from approximate waveform extraction from an ECG image "
                "and should be confirmed with raw signal data when available."
            )

        return (
            f"{base_reason} The strongest matching class is {display_name}. "
            f"{level_sentence} Class scores: {score_summary}.{source_sentence}"
        )

    @staticmethod
    def _base_clinical_reason(predicted_label: str, confidence_phrase: str) -> str:
        """Return a class-specific clinical explanation sentence."""

        if predicted_label == "NORM":
            return (
                "The ECG waveform appears normal with stable rhythm patterns "
                f"and {confidence_phrase}."
            )
        if predicted_label in {"MI", "AMI", "STEMI"}:
            return (
                "The ECG indicates patterns consistent with myocardial infarction, "
                f"which is a high-risk condition, with {confidence_phrase}."
            )
        if predicted_label == "STTC":
            return (
                "The ECG shows ST/T waveform changes that may reflect repolarization "
                f"abnormality, with {confidence_phrase}."
            )
        if predicted_label == "CD":
            return (
                "The ECG shows conduction-related waveform changes suggesting possible "
                f"conduction disturbance, with {confidence_phrase}."
            )
        if predicted_label == "HYP":
            return (
                "The ECG shows voltage and waveform patterns that may be consistent with "
                f"hypertrophy, with {confidence_phrase}."
            )
        return (
            f"The ECG most closely matches the {DISPLAY_LABEL_MAP.get(predicted_label, predicted_label)} "
            f"pattern, with {confidence_phrase}."
        )

    @staticmethod
    def _confidence_phrase(confidence: float) -> str:
        """Convert a confidence score into a readable phrase."""

        if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return f"high confidence (score {confidence:.2f})"
        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            return f"moderate confidence (score {confidence:.2f})"
        return f"low confidence (score {confidence:.2f})"

    @staticmethod
    def _level_adjustment_reason(
        clinical_level: str,
        final_level: str,
        confidence: float,
    ) -> str:
        """Explain whether confidence changed the final alert level."""

        if final_level != clinical_level:
            return (
                f"The baseline clinical level is {clinical_level}, but the final Level is "
                f"{final_level} because the confidence score is {confidence:.2f}."
            )
        return (
            f"The final Level remains {final_level} because the confidence score is "
            f"{confidence:.2f}."
        )

    def _format_score_summary(self, probabilities: List[float]) -> str:
        """Format class probabilities into a stable readable summary."""

        score_parts: List[str] = []
        for class_index, class_label in enumerate(self.class_labels):
            class_name = DISPLAY_LABEL_MAP.get(class_label, class_label)
            class_score = probabilities[class_index] if class_index < len(probabilities) else 0.0
            score_parts.append(f"{class_name} {class_score:.3f}")
        return ", ".join(score_parts)
