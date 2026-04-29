"""
Echo Agent — Week 6
====================
Wraps the Echo model into an independent, callable agent.

Responsibilities:
    - Accept real video path OR dummy/simulated input
    - Validate all inputs before passing to model
    - Return standardized output dict
    - Handle errors gracefully (never crash the pipeline)

Output Format (always):
    {
        "level":  "Low" | "Medium" | "High" | "Unknown",
        "score":  float (0.0 – 1.0),
        "source": "real" | "dummy",
        "error":  str | None
    }
"""

import os
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
VALID_LEVELS   = {"Low", "Medium", "High"}
DUMMY_DEFAULT  = {"level": "Low", "score": 0.5, "source": "dummy", "error": None}


# ── Internal helpers ───────────────────────────────────────────────────────

def _validate_video_path(video_path):
    """
    Checks whether the given path is a valid, readable video file.

    Returns
    -------
    (bool, str | None)
        (True, None)         — path is valid
        (False, reason_str)  — path is invalid, with explanation
    """
    if not isinstance(video_path, str):
        return False, f"video_path must be a string, got {type(video_path).__name__}"

    if not video_path.strip():
        return False, "video_path is an empty string"

    if not os.path.exists(video_path):
        return False, f"File not found: {video_path}"

    if not os.path.isfile(video_path):
        return False, f"Path is not a file: {video_path}"

    ext = os.path.splitext(video_path)[1].lower()
    allowed_ext = {".avi", ".mp4", ".mov", ".mkv"}
    if ext not in allowed_ext:
        return False, f"Unsupported file extension '{ext}'. Expected one of {allowed_ext}"

    if os.path.getsize(video_path) == 0:
        return False, f"File is empty: {video_path}"

    return True, None


def _build_output(level, score, source="real", error=None):
    """
    Constructs a standardised output dict and clamps/validates values.
    """
    # Clamp score to [0.0, 1.0]
    score = float(np.clip(score, 0.0, 1.0))

    # Fallback level if something unexpected came from the model
    if level not in VALID_LEVELS:
        level  = "Unknown"
        error  = error or f"Unexpected level value received from model: '{level}'"

    return {
        "level":  level,
        "score":  round(score, 4),
        "source": source,
        "error":  error,
    }


# ── Public Agent Function ──────────────────────────────────────────────────

def echo_agent(input_data):
    """
    Main entry point for the Echo Agent.

    Parameters
    ----------
    input_data : str | None | dict
        - str  : path to an echocardiography video file (.avi, .mp4, etc.)
        - None : triggers dummy mode (returns a safe default prediction)
        - dict : reserved for future structured/real-time input

    Returns
    -------
    dict
        {
            "level":  "Low" | "Medium" | "High" | "Unknown",
            "score":  float,
            "source": "real" | "dummy",
            "error":  str | None
        }

    Examples
    --------
    Real input:
        >>> result = echo_agent("patient_001_echo.avi")
        >>> print(result)
        {"level": "High", "score": 0.812, "source": "real", "error": None}

    Dummy input (no video available yet):
        >>> result = echo_agent(None)
        >>> print(result)
        {"level": "Low", "score": 0.5, "source": "dummy", "error": None}
    """

    # ── 1. Handle None / dummy mode ────────────────────────────────────────
    if input_data is None:
        return dict(DUMMY_DEFAULT)

    # ── 2. Handle future dict-based structured input ───────────────────────
    #    (Placeholder for real-time EF values or DICOM metadata)
    if isinstance(input_data, dict):
        ef_value = input_data.get("ef_value")
        if ef_value is not None:
            return _predict_from_ef(float(ef_value))
        # Unknown dict structure — fall through to dummy
        return _build_output(
            "Unknown", 0.0, source="dummy",
            error="Dict input received but 'ef_value' key not found. Returning dummy."
        )

    # ── 3. Video file path — validate first ───────────────────────────────
    is_valid, reason = _validate_video_path(input_data)

    if not is_valid:
        # Do NOT crash the pipeline — return Unknown with error note
        return _build_output(
            "Unknown", 0.0, source="dummy",
            error=f"Input validation failed: {reason}"
        )

    # ── 4. Run the real model ──────────────────────────────────────────────
    try:
        from echo_module.echo_model import predict as echo_predict
        raw = echo_predict(input_data)

        level = raw.get("level", "Unknown")
        score = raw.get("score", 0.0)

        return _build_output(level, score, source="real")

    except ImportError as e:
        return _build_output(
            "Unknown", 0.0, source="dummy",
            error=f"Model import failed (dependencies missing?): {e}"
        )
    except Exception as e:
        return _build_output(
            "Unknown", 0.0, source="dummy",
            error=f"Model prediction error: {e}"
        )


# ── EF-value shortcut (for future structured input) ───────────────────────

def _predict_from_ef(ef_value):
    """
    Converts a raw Ejection Fraction (EF) percentage directly to a risk level,
    bypassing the video model.  Useful when EF is already known (e.g., from DICOM).

    EF mapping (mirrors final_echo_model.py LABEL_MAP + RISK_MAP):
        EF > 55  → Normal  → Low
        EF 40-55 → Mild    → Low
        EF 30-40 → Moderate→ Medium
        EF < 30  → Severe  → High
    """
    if ef_value > 55:
        level, score = "Low",    0.9
    elif ef_value > 40:
        level, score = "Low",    0.65
    elif ef_value > 30:
        level, score = "Medium", 0.70
    else:
        level, score = "High",   0.85

    return _build_output(level, score, source="real")


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Echo Agent — Test Suite")
    print("=" * 50)

    # Test 1: Dummy mode
    print("\n[Test 1] Dummy mode (input=None)")
    result = echo_agent(None)
    print("  Result:", result)
    assert result["source"] == "dummy"
    assert result["error"] is None

    # Test 2: Missing file
    print("\n[Test 2] Non-existent video path")
    result = echo_agent("non_existent_video.avi")
    print("  Result:", result)
    assert result["level"] == "Unknown"
    assert result["error"] is not None

    # Test 3: Bad type
    print("\n[Test 3] Wrong input type (integer)")
    result = echo_agent(42)
    print("  Result:", result)
    assert result["error"] is not None

    # Test 4: EF-based input
    print("\n[Test 4] EF-value dict input — Severe (EF=25)")
    result = echo_agent({"ef_value": 25})
    print("  Result:", result)
    assert result["level"] == "High"

    print("\n[Test 5] EF-value dict input — Normal (EF=60)")
    result = echo_agent({"ef_value": 60})
    print("  Result:", result)
    assert result["level"] == "Low"

    # Test 6: Real file (only if sample exists)
    sample = "sample_echo.avi"
    if os.path.exists(sample):
        print(f"\n[Test 6] Real video: {sample}")
        result = echo_agent(sample)
        print("  Result:", result)
        assert result["source"] == "real"
    else:
        print(f"\n[Test 6] Skipped — {sample} not present")

    print("\n✅ All tests passed.")
