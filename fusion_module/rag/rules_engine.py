def _level_to_severity(level):
    """Map risk level string to a numeric severity (0=Low, 1=Medium, 2=High, -1=Unknown)."""
    return {"Low": 0, "Medium": 1, "High": 2}.get(level, -1)


def _confidence_label(score):
    """Convert a 0-1 score to a readable confidence phrase."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "unknown confidence"
    if score >= 0.75:
        return f"high confidence ({score * 100:.1f}%)"
    if score >= 0.50:
        return f"moderate confidence ({score * 100:.1f}%)"
    return f"low confidence ({score * 100:.1f}%)"


def _build_summary(echo_level, ecg_level, clinical_level, risk_percentage):
    """Generate a detailed multi-modality narrative summary."""
    levels = {
        "Echo (Structural)": echo_level,
        "ECG (Electrical)": ecg_level,
        "Clinical (Patient Data)": clinical_level,
    }

    high_sources = [k for k, v in levels.items() if v == "High"]
    medium_sources = [k for k, v in levels.items() if v == "Medium"]
    low_sources = [k for k, v in levels.items() if v == "Low"]
    unavailable = [k for k, v in levels.items() if v not in ("Low", "Medium", "High")]

    try:
        risk_pct = float(risk_percentage)
    except (TypeError, ValueError):
        risk_pct = None

    risk_str = f"{risk_pct:.1f}%" if risk_pct is not None else "N/A"

    if high_sources and len(high_sources) >= 2:
        base = (
            f"CRITICAL: Multiple high-risk signals detected across {', '.join(high_sources)}. "
            f"The fused risk score is {risk_str}, indicating a serious cardiovascular concern "
            f"that warrants immediate medical review."
        )
    elif high_sources:
        base = (
            f"HIGH RISK detected via {high_sources[0]}. "
            f"The fused risk score is {risk_str}. "
            f"Immediate clinical evaluation is strongly recommended."
        )
    elif medium_sources:
        base = (
            f"MODERATE RISK: Elevated signals found in {', '.join(medium_sources)}. "
            f"The fused risk score is {risk_str}. "
            f"Close monitoring and lifestyle review are advised."
        )
    else:
        base = (
            f"LOW RISK: All evaluated modalities show no major abnormalities. "
            f"The fused risk score is {risk_str}. "
            f"Routine check-ups are recommended to maintain good heart health."
        )

    if unavailable:
        base += f" Note: {', '.join(unavailable)} modality result(s) were unavailable."

    return base


def _build_recommendations(echo_level, ecg_level, clinical_level):
    """Return actionable clinical recommendations based on risk levels."""
    recs = []
    max_severity = max(
        _level_to_severity(echo_level),
        _level_to_severity(ecg_level),
        _level_to_severity(clinical_level),
    )

    if max_severity == 2:
        recs += [
            "Seek immediate cardiology consultation",
            "Do not engage in strenuous physical activity until cleared by a doctor",
            "Monitor blood pressure and heart rate continuously",
            "Review and adjust any existing cardiac medications with your physician",
        ]
    elif max_severity == 1:
        recs += [
            "Schedule a follow-up with your cardiologist within 2-4 weeks",
            "Adopt a heart-healthy diet (reduce sodium, saturated fats)",
            "Engage in moderate aerobic activity as tolerated (30 min/day, 5 days/week)",
            "Monitor cholesterol and glucose levels regularly",
        ]
    else:
        recs += [
            "Maintain current healthy lifestyle habits",
            "Annual cardiac screening recommended",
            "Stay physically active and maintain a balanced diet",
        ]

    return recs


def apply_rules(echo, ecg, clinical, fusion_output):
    """
    Apply logical rules to generate a rich, structured clinical reasoning output.
    """
    echo_level = echo.get("level")
    ecg_level = ecg.get("level")
    clinical_level = clinical.get("level")
    risk_percentage = fusion_output.get("risk_percentage")

    summary = _build_summary(echo_level, ecg_level, clinical_level, risk_percentage)
    recommendations = _build_recommendations(echo_level, ecg_level, clinical_level)

    details = []

    # ── Structural (Echo) ──────────────────────────────────────────────────
    echo_conf = _confidence_label(echo.get("score"))
    echo_reason = echo.get("reason") or "No details available"
    details.append(
        f"[Echo – Structural] Risk Level: {echo_level or 'N/A'} | {echo_conf}. {echo_reason}"
    )

    # ── Electrical (ECG) ───────────────────────────────────────────────────
    ecg_conf = _confidence_label(ecg.get("score"))
    ecg_reason = ecg.get("reason") or "No details available"
    details.append(
        f"[ECG – Electrical]  Risk Level: {ecg_level or 'N/A'} | {ecg_conf}. {ecg_reason}"
    )

    # ── Clinical (Patient Data) ────────────────────────────────────────────
    clin_conf = _confidence_label(clinical.get("score"))
    clin_reason = clinical.get("reason") or "No details available"
    details.append(
        f"[Clinical – Patient] Risk Level: {clinical_level or 'N/A'} | {clin_conf}. {clin_reason}"
    )

    # ── Recommendations ────────────────────────────────────────────────────
    details.append("Recommendations:")
    for rec in recommendations:
        details.append(f"  • {rec}")

    return {"summary": summary, "details": details}