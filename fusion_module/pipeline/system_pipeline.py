import textwrap

from clinical_module.agent.clinical_agent import clinical_agent, validate_clinical_input
from echo_module.agent.echo_agent import echo_agent
from ecg_module.agent.ecg_agent import predict_ecg_signal as ecg_agent
from fusion_module.models.fusion_model import FusionModel
from fusion_module.rag.rag_pipeline import run_rag


REPORT_WIDTH = 30
LABEL_WIDTH = 18


def _prompt_int(prompt: str) -> int:
    """Read an integer value from the CLI, retrying on invalid input."""

    while True:
        raw_value = input(prompt).strip()
        try:
            return int(raw_value)
        except ValueError:
            print("Enter a whole number.")


def _prompt_float(prompt: str) -> float:
    """Read a numeric value from the CLI, retrying on invalid input."""

    while True:
        raw_value = input(prompt).strip()
        try:
            return float(raw_value)
        except ValueError:
            print("Enter a numeric value.")


def _prompt_clinical_input() -> dict:
    """Collect clinical values and re-prompt until they pass validation."""

    while True:
        print("\nEnter clinical data:")
        clinical_input = {
            "age": _prompt_int("Age (years): "),
            "gender": _prompt_int("Gender (1=Female, 2=Male): "),
            "height": _prompt_float("Height (cm): "),
            "weight": _prompt_float("Weight (kg): "),
            "ap_hi": _prompt_int("Systolic BP (mmHg): "),
            "ap_lo": _prompt_int("Diastolic BP (mmHg): "),
            "cholesterol": _prompt_int(
                "Cholesterol (1=normal, 2=above normal, 3=well above normal): "
            ),
            "gluc": _prompt_int(
                "Glucose (1=normal, 2=above normal, 3=well above normal): "
            ),
            "smoke": _prompt_int("Smoke (0=No, 1=Yes): "),
            "alco": _prompt_int("Alcohol (0=No, 1=Yes): "),
            "active": _prompt_int("Active (0=No, 1=Yes): "),
        }

        errors = validate_clinical_input(clinical_input)
        if not errors:
            return clinical_input

        print("\nClinical input has validation errors:")
        for error in errors:
            print(f"- {error}")
        print("Please re-enter the clinical values.")


def _prompt_required_ecg_path() -> str:
    """Require a user-supplied ECG file path instead of using dummy input."""

    while True:
        ecg_path = input("\nEnter ECG file path (.csv or ECG image): ").strip()
        if ecg_path:
            return ecg_path
        print("ECG input is required. Provide a .csv file or an ECG image path.")


def _print_section_header(title: str) -> None:
    """Print a consistent report section heading."""

    print()
    print("=" * REPORT_WIDTH)
    print(title)
    print("=" * REPORT_WIDTH)
    print()


def _print_field(label: str, value: object) -> None:
    """Print a wrapped, aligned label/value row."""

    text = str(value) if value not in (None, "") else "N/A"
    prefix = f"  {label:<{LABEL_WIDTH}}: "
    wrapped = textwrap.fill(
        text,
        width=96,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    print(wrapped)


def _print_list_field(label: str, items: list[str]) -> None:
    """Print a labeled list with clean indentation."""

    prefix = f"  {label:<{LABEL_WIDTH}}: "
    if not items:
        print(f"{prefix}N/A")
        return

    continuation_indent = " " * len(prefix)
    for index, item in enumerate(items):
        initial_indent = prefix if index == 0 else continuation_indent
        wrapped = textwrap.fill(
            item,
            width=96,
            initial_indent=initial_indent,
            subsequent_indent=continuation_indent + "  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        print(wrapped)


def _format_prediction(level: object) -> str:
    """Convert a raw level into a readable prediction label."""

    if level in {"Low", "Medium", "High"}:
        return f"{level} Risk"
    if level == "Unknown":
        return "Unknown"
    return "Unavailable"


def _format_confidence(score: object) -> str:
    """Render model score as a percentage when possible."""

    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric_score * 100:.2f}%"


def _first_non_empty(*values: object) -> str:
    """Return the first meaningful string-like value."""

    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "No additional details provided."


def _print_clinical_section(clinical_output: dict) -> None:
    """Print the Clinical Agent section."""

    _print_section_header("CLINICAL AGENT RESULT")
    _print_field("Prediction", _format_prediction(clinical_output.get("level")))
    _print_field("Risk Level", clinical_output.get("level"))
    _print_field("Confidence", _format_confidence(clinical_output.get("score")))
    _print_field(
        "Important Factors",
        _first_non_empty(clinical_output.get("reason")),
    )


def _print_ecg_section(ecg_output: dict) -> None:
    """Print the ECG Agent section."""

    _print_section_header("ECG AGENT RESULT")
    _print_field("Prediction", _format_prediction(ecg_output.get("level")))
    _print_field("Risk Level", ecg_output.get("level"))
    _print_field("Confidence", _format_confidence(ecg_output.get("score")))
    _print_field(
        "Abnormalities",
        _first_non_empty(ecg_output.get("reason")),
    )


def _print_echo_section(echo_output: dict) -> None:
    """Print the Echo Agent section."""

    _print_section_header("ECHO AGENT RESULT")
    _print_field("Prediction", _format_prediction(echo_output.get("level")))
    _print_field("Risk Level", echo_output.get("level"))
    _print_field("Confidence", _format_confidence(echo_output.get("score")))
    _print_field(
        "Findings",
        _first_non_empty(
            echo_output.get("reason"),
            echo_output.get("error"),
            f"Echo inference completed using {echo_output.get('source', 'unknown')} input.",
        ),
    )


def _print_fusion_section(result: dict) -> None:
    """Print the final fused system result."""

    fusion_output = result["fusion"]
    rag_output = result["rag"]

    _print_section_header("FINAL SYSTEM RESULT")
    _print_field("Risk Level", fusion_output.get("final_level"))
    risk_percentage = fusion_output.get("risk_percentage")
    if risk_percentage in (None, ""):
        formatted_percentage = "N/A"
    else:
        formatted_percentage = f"{risk_percentage}%"
    _print_field("Risk Percentage", formatted_percentage)
    _print_field("Summary", rag_output.get("explanation"))
    detail_items = [f"- {item}" for item in rag_output.get("details", [])]
    _print_list_field("Details", detail_items)


def _print_report(result: dict) -> None:
    """Print the pipeline result in a clean professional layout."""

    _print_section_header("HEART DISEASE SYSTEM REPORT")
    _print_clinical_section(result["clinical"])
    _print_ecg_section(result["ecg"])
    _print_echo_section(result["echo"])
    _print_fusion_section(result)


class SystemPipeline:
    def __init__(self):
        self.fusion = FusionModel()

    def run(self, echo_input, ecg_input, clinical_input):
        echo_output = echo_agent(echo_input)
        ecg_output = ecg_agent(ecg_input)
        clinical_output = clinical_agent(clinical_input)

        fusion_output = self.fusion.predict(
            echo_output,
            ecg_output,
            clinical_output,
        )

        rag_output = run_rag(
            {
                "echo": echo_output,
                "ecg": ecg_output,
                "clinical": clinical_output,
                "fusion": fusion_output,
            }
        )

        return {
            "echo": echo_output,
            "ecg": ecg_output,
            "clinical": clinical_output,
            "fusion": fusion_output,
            "rag": rag_output,
        }


if __name__ == "__main__":
    pipeline = SystemPipeline()

    print("\n===== HEART DISEASE SYSTEM  =====\n")

    clinical_input = _prompt_clinical_input()
    echo_input = input("\nEnter Echo video path (press Enter to skip): ").strip()
    ecg_input = _prompt_required_ecg_path()

    result = pipeline.run(echo_input, ecg_input, clinical_input)
    _print_report(result)
