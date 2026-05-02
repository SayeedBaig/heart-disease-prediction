from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_module.agent.ecg_agent import ECGAgent, ECGPredictionResponse
from ecg_module.model.ecg_model_loader import ModelLoadingError


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CSV and image ECG testing."""

    parser = argparse.ArgumentParser(
        description="Test the ECG agent with CSV and/or image inputs."
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to a CSV ECG file with shape (1000, 12).",
    )
    parser.add_argument(
        "--image-file",
        type=str,
        default=None,
        help="Path to an ECG image file for approximate waveform extraction.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to the trained .pth checkpoint.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for the test runner.",
    )
    args = parser.parse_args()
    if not args.csv_file and not args.image_file:
        parser.error("Provide at least one of --csv-file or --image-file.")
    return args


def configure_logging(level: str) -> None:
    """Configure application logging for CLI execution."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def print_result(title: str, result: ECGPredictionResponse) -> None:
    """Print a labeled ECG prediction response."""

    print(f"{title}:")
    print(json.dumps(result, indent=4))
    print()


def is_error_response(result: Dict[str, object]) -> bool:
    """Detect whether a response represents an operational failure."""

    reason = str(result.get("Reason", ""))
    return reason.startswith("Prediction failed:")


def main() -> int:
    """Run the ECG agent for the requested CSV and/or image inputs."""

    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        agent = ECGAgent(model_path=args.model_path)
    except ModelLoadingError as exc:
        logger.exception("Unable to initialize ECG agent.")
        result: ECGPredictionResponse = {
            "Level": "Low",
            "Score": 0.0,
            "Reason": f"Prediction failed: {exc}",
        }
        print(json.dumps(result, indent=4))
        return 1

    exit_code = 0

    if args.csv_file:
        csv_result = agent.predict_from_csv(args.csv_file)
        print_result("CSV Prediction", csv_result)
        if is_error_response(csv_result):
            exit_code = 1

    if args.image_file:
        image_result = agent.predict_from_image(args.image_file)
        print_result("Image Prediction", image_result)
        if is_error_response(image_result):
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
