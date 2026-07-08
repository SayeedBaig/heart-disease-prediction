from reportlab.lib.enums import TA_JUSTIFY
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
NAVY       = colors.HexColor("#0D1B2A")
TEAL       = colors.HexColor("#1A7A8A")
LIGHT_TEAL = colors.HexColor("#E8F4F6")
SLATE      = colors.HexColor("#4A5568")
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor("#F7F9FA")
MID_GRAY   = colors.HexColor("#CBD5E0")
BLACK      = colors.black

# Page content width (A4 minus margins)
CONTENT_W = 170 * mm
COL_LABEL = 58 * mm
COL_VALUE = CONTENT_W - COL_LABEL


# ---------------------------------------------------------------------------
# Style registry  (built once per call to avoid duplicate-name warnings)
# ---------------------------------------------------------------------------
def build_styles() -> Dict[str, ParagraphStyle]:
    return {
        "brand": ParagraphStyle(
            "brand", fontName="Helvetica-Bold", fontSize=24,
            textColor=NAVY, alignment=TA_CENTER, spaceAfter=0,
            spaceBefore=0, leading=28,
        ),
        "brand_sub": ParagraphStyle(
            "brand_sub", fontName="Helvetica", fontSize=10,
            textColor=TEAL, alignment=TA_CENTER, spaceAfter=0,
            leading=14,
        ),
        "tagline": ParagraphStyle(
            "tagline", fontName="Helvetica-Oblique", fontSize=9,
            textColor=SLATE, alignment=TA_CENTER, spaceAfter=4,
        ),
        "section_title": ParagraphStyle(
            "section_title", fontName="Helvetica-Bold", fontSize=11,
            textColor=WHITE, alignment=TA_LEFT,
            spaceAfter=0, spaceBefore=0,
        ),
        "sub_section": ParagraphStyle(
            "sub_section", fontName="Helvetica-Bold", fontSize=10,
            textColor=NAVY, spaceBefore=6, spaceAfter=2,
        ),
        "kv_key": ParagraphStyle(
            "kv_key", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY,
        ),
        "kv_val": ParagraphStyle(
            "kv_val", fontName="Helvetica", fontSize=9, textColor=SLATE,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=9,
            textColor=SLATE, leftIndent=8, spaceAfter=3,
        ),
        "plain": ParagraphStyle(
            "plain", fontName="Helvetica", fontSize=9,
            textColor=SLATE, spaceAfter=4, leading=13,
            alignment=TA_JUSTIFY,
        ),
        "risk_box_label": ParagraphStyle(
            "risk_box_label", fontName="Helvetica-Bold", fontSize=11,
            textColor=SLATE, alignment=TA_CENTER,
        ),
        "risk_box_value": ParagraphStyle(
            "risk_box_value", fontName="Helvetica-Bold", fontSize=22,
            textColor=BLACK, alignment=TA_CENTER, leading=26,
        ),
        "ref_header": ParagraphStyle(
            "ref_header", fontName="Helvetica-Bold", fontSize=9,
            textColor=NAVY,
        ),
        "ref_body": ParagraphStyle(
            "ref_body", fontName="Helvetica", fontSize=8,
            textColor=SLATE, leading=11,
        ),
        "footer": ParagraphStyle(
            "footer", fontName="Helvetica-Oblique", fontSize=8,
            textColor=MID_GRAY, alignment=TA_CENTER,
        ),
        "badge": ParagraphStyle(
            "badge", fontName="Helvetica-Bold", fontSize=12,
            textColor=WHITE, alignment=TA_CENTER,
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scalar(value: Any) -> str:
    """Convert any scalar to a clean, human-readable string."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        # percentages stored as 0–1
        if 0.0 <= value <= 1.0:
            return f"{value:.1%}"
        return f"{value:.2f}"
    return str(value)


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _label(key: str) -> str:
    return key.replace("_", " ").title()


# ---------------------------------------------------------------------------
# PdfFormatter — all formatting logic, returns list[Flowable]
# ---------------------------------------------------------------------------
class PdfFormatter:

    def __init__(self, styles: Dict[str, ParagraphStyle]) -> None:
        self.s = styles

    # ── Primitives ───────────────────────────────────────────────────────

    def spacer(self, h: int = 4) -> Spacer:
        return Spacer(1, h * mm)

    def hr(self, thickness: float = 0.5, color=MID_GRAY) -> HRFlowable:
        return HRFlowable(
            width="100%", thickness=thickness, color=color,
            spaceAfter=3, spaceBefore=3,
        )

    def section_banner(self, title: str) -> Table:
        """Teal full-width banner used as section heading."""
        t = Table(
            [[Paragraph(title.upper(), self.s["section_title"])]],
            colWidths=[CONTENT_W],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), TEAL),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        return t

    # ── KV table ─────────────────────────────────────────────────────────

    def kv_table(self, data: Dict[str, Any]) -> Optional[Table]:
        """Two-column label / value table, skipping empty values."""
        rows = []
        for key, val in data.items():
            if _is_empty(val):
                continue
            if isinstance(val, (dict, list)):
                continue          # handled by caller with smarter renderer
            rows.append([
                Paragraph(_label(key), self.s["kv_key"]),
                Paragraph(_scalar(val), self.s["kv_val"]),
            ])
        if not rows:
            return None
        t = Table(rows, colWidths=[COL_LABEL, COL_VALUE])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    # ── Bullet list ───────────────────────────────────────────────────────

    def bullet_list(self, items: list) -> List:
        out = []
        for item in items:
            if _is_empty(item):
                continue
            text = item if isinstance(item, str) else _scalar(item)
            out.append(Paragraph(f"&#8226;&#160;&#160;{text}", self.s["bullet"]))
        return out

    # ── Smart dict block ──────────────────────────────────────────────────

    def format_dict(self, data: Dict[str, Any]) -> List:
        """
        Renders a dict as a kv_table.
        Nested dicts/lists are rendered recursively with a sub-heading.
        """
        if _is_empty(data):
            return []
        out = []
        t = self.kv_table(data)
        if t:
            out.append(t)
        # handle any nested dicts or lists
        for key, val in data.items():
            if isinstance(val, dict) and val:
                out.append(Paragraph(_label(key), self.s["sub_section"]))
                out.extend(self.format_dict(val))
            elif isinstance(val, list) and val:
                out.append(Paragraph(_label(key), self.s["sub_section"]))
                out.extend(self.format_list(val))
        return out

    def format_list(self, items: list) -> List:
        """Detects list[str] vs list[dict] and renders appropriately."""
        if not items:
            return []
        if all(isinstance(i, str) for i in items):
            return self.bullet_list(items)
        # list[dict] — render each as a mini kv block
        out = []
        for i, item in enumerate(items, 1):
            if isinstance(item, dict):
                t = self.kv_table(item)
                if t:
                    out.append(t)
                    out.append(self.spacer(2))
            elif not _is_empty(item):
                out.append(Paragraph(f"&#8226;&#160;&#160;{_scalar(item)}", self.s["bullet"]))
        return out

    # ── Patient information ───────────────────────────────────────────────

    def format_patient_info(self, patient: Dict[str, Any]) -> List:
        if not patient:
            return []
        FIELD_ORDER = [
            "patient_id", "full_name", "gender",
            "date_of_birth", "email", "phone",
        ]
        ordered = {k: patient[k] for k in FIELD_ORDER if k in patient}
        remainder = {k: v for k, v in patient.items() if k not in FIELD_ORDER}
        ordered.update(remainder)
        out = [self.section_banner("Patient Information"), self.spacer(2)]
        t = self.kv_table(ordered)
        if t:
            out.append(t)
        return out

    # ── Risk summary (black/white box) ────────────────────────────────────

    def format_risk_summary(self, level: str, percentage: Any) -> List:
        if not level:
            return []

        level_text = str(level).upper()
        pct_text   = _scalar(percentage) if percentage is not None else None

        inner_rows = [[Paragraph(level_text, self.s["risk_box_value"])]]
        if pct_text:
            inner_rows.append([Paragraph(pct_text, self.s["risk_box_label"])])

        box = Table(inner_rows, colWidths=[CONTENT_W * 0.5])
        box.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("BOX",           (0, 0), (-1, -1), 1.5, BLACK),
            ("LINEABOVE",     (0, 1), (-1, 1),  0.5, MID_GRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))

        # Centre the box on the page
        wrapper = Table([[box]], colWidths=[CONTENT_W])
        wrapper.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        return [self.section_banner("Risk Assessment"), self.spacer(4), wrapper, self.spacer(2)]

    # ── Clinical analysis ─────────────────────────────────────────────────

    def format_clinical_analysis(self, data: Dict[str, Any]) -> List:
        if _is_empty(data):
            return []
        PRIORITY = ["risk_level", "confidence", "key_findings",
                    "recommendation", "reason"]
        ordered = {k: data[k] for k in PRIORITY if k in data}
        ordered.update({k: v for k, v in data.items() if k not in PRIORITY})
        return [self.section_banner("Clinical Analysis"),
                self.spacer(2)] + self.format_dict(ordered)

    # ── ECG / Echo — hides empty / dummy fields ───────────────────────────

    _NOISE_VALUES   = {"dummy", "n/a", "none", "error", "—", "-", "null", "0.0%"}
    _NOISE_PREFIXES = ("invalid", "dummy", "error:", "provide a file", "no video",
                       "none provided", "not provided")

    def format_ecg_echo(self, data: Dict[str, Any], label: str) -> List:
        if _is_empty(data):
            return []
        cleaned = {}
        for k, v in data.items():
            if _is_empty(v):
                continue
            v_str = str(v).strip().lower()
            if v_str in self._NOISE_VALUES:
                continue
            if any(v_str.startswith(p) for p in self._NOISE_PREFIXES):
                continue
            cleaned[k] = v
        if not cleaned:
            return []
        return [self.section_banner(label), self.spacer(2)] + self.format_dict(cleaned)

    # ── Digital twin ──────────────────────────────────────────────────────

    def format_digital_twin(self, data: Any) -> List:
        if _is_empty(data):
            return []
        out = [self.section_banner("Digital Twin Simulation"), self.spacer(2)]

        # Scenario list: [{"scenario":..., "risk":..., "improvement":...}, ...]
        scenarios = None
        if isinstance(data, list):
            scenarios = data
        elif isinstance(data, dict):
            # Try to find a list of scenarios inside
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    scenarios = v
                    break
            if not scenarios:
                # Flat dict — just render as kv
                out.extend(self.format_dict(data))
                return out

        if scenarios:
            header = [
                Paragraph("<b>Scenario</b>",   self.s["kv_key"]),
                Paragraph("<b>Risk</b>",        self.s["kv_key"]),
                Paragraph("<b>Improvement</b>", self.s["kv_key"]),
            ]
            rows = [header]
            for item in scenarios:
                if not isinstance(item, dict):
                    continue
                scenario    = str(item.get("scenario", item.get("name", "—")))
                risk_raw    = item.get("risk", item.get("risk_score", "—"))
                improvement = item.get("improvement", item.get("change", item.get("delta", "—")))

                # Format risk as percentage
                try:
                    r = float(risk_raw)
                    risk_str = f"{r:.1f}%" if r > 1 else f"{r:.1%}"
                except (TypeError, ValueError):
                    risk_str = str(risk_raw)

                # Format improvement as ↓X%
                try:
                    imp = float(improvement)
                    if imp > 1:          # already in percentage points
                        imp_str = f"&#8595;{abs(imp):.1f}%"
                    elif imp != 0:
                        imp_str = f"&#8595;{abs(imp):.1%}"
                    else:
                        imp_str = "—"
                except (TypeError, ValueError):
                    imp_str = str(improvement)
                rows.append([
                    Paragraph(scenario,  self.s["kv_val"]),
                    Paragraph(risk_str,  self.s["kv_val"]),
                    Paragraph(imp_str,   self.s["kv_val"]),
                ])
            col3 = CONTENT_W / 3
            t = Table(rows, colWidths=[col3 * 1.4, col3 * 0.8, col3 * 0.8])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ]))
            out.append(t)
        return out

    # ── Medical explanation ───────────────────────────────────────────────

    def format_medical_explanation(self, data: Any) -> List:
        if _is_empty(data):
            return []
        out = [self.section_banner("Medical Explanation"), self.spacer(2)]

        if isinstance(data, str):
            out.append(Paragraph(data, self.s["plain"]))
            return out

        if not isinstance(data, dict):
            return out

        for sub_key in ["summary", "details"]:
            val = data.get(sub_key)
            if val and isinstance(val, str):
                out.append(Paragraph(_label(sub_key), self.s["sub_section"]))
                out.append(Paragraph(val, self.s["plain"]))

        for sub_key in ["recommendations", "lifestyle_suggestions"]:
            items = data.get(sub_key, [])
            if items:
                out.append(Paragraph(_label(sub_key), self.s["sub_section"]))
                if isinstance(items, list):
                    out.extend(self.bullet_list(
                        [i.get("text", str(i)) if isinstance(i, dict) else str(i)
                         for i in items]
                    ))
                else:
                    out.append(Paragraph(str(items), self.s["plain"]))

        # remaining keys
        handled = {"summary", "details", "recommendations", "lifestyle_suggestions"}
        for k, v in data.items():
            if k in handled or _is_empty(v):
                continue
            out.append(Paragraph(_label(k), self.s["sub_section"]))
            if isinstance(v, list):
                out.extend(self.format_list(v))
            elif isinstance(v, dict):
                t = self.kv_table(v)
                if t:
                    out.append(t)
            else:
                out.append(Paragraph(str(v), self.s["plain"]))

        return out

    # ── AI recommendation ─────────────────────────────────────────────────

    def format_ai_recommendation(self, data: Any) -> List:
        if _is_empty(data):
            return []
        out = [self.section_banner("AI Recommendation"), self.spacer(2)]

        if isinstance(data, str):
            out.append(Paragraph(data, self.s["plain"]))
            return out

        if not isinstance(data, dict):
            return out

        for sub_key in ["overall_assessment", "assessment", "summary"]:
            val = data.get(sub_key)
            if val:
                out.append(Paragraph("Overall Assessment", self.s["sub_section"]))
                out.append(Paragraph(str(val), self.s["plain"]))
                break

        evidence = data.get("evidence")
        if evidence:
            out.append(Paragraph("Evidence", self.s["sub_section"]))
            if isinstance(evidence, list):
                out.extend(self.bullet_list([str(e) for e in evidence]))
            else:
                out.append(Paragraph(str(evidence), self.s["plain"]))

        recs = data.get("recommendations", [])
        if recs:
            out.append(Paragraph("Recommendations", self.s["sub_section"]))
            if isinstance(recs, list):
                out.extend(self.bullet_list(
                    [r.get("text", str(r)) if isinstance(r, dict) else str(r)
                     for r in recs]
                ))
            else:
                out.append(Paragraph(str(recs), self.s["plain"]))

        return out

    # ── Supporting references ─────────────────────────────────────────────

    MAX_EXCERPT = 300   # characters shown per reference

    def format_references(self, chunks: list) -> List:
        if not chunks:
            return []
        out = [self.section_banner("Supporting References"), self.spacer(2)]

        for i, chunk in enumerate(chunks, 1):
            if isinstance(chunk, dict):
                source  = chunk.get("source", chunk.get("title", f"Reference {i}"))
                score   = chunk.get("score", chunk.get("similarity", chunk.get("relevance")))
                page    = chunk.get("page", chunk.get("page_number"))
                content = chunk.get("content", chunk.get("text", ""))

                # Truncate content to a short excerpt
                excerpt = str(content).strip()
                if len(excerpt) > self.MAX_EXCERPT:
                    excerpt = excerpt[:self.MAX_EXCERPT].rsplit(" ", 1)[0] + " …"

                meta_parts = [f"[{i}]  {source}"]
                if score is not None:
                    try:
                        meta_parts.append(f"Relevance: {float(score):.0%}")
                    except (ValueError, TypeError):
                        pass
                if page is not None:
                    meta_parts.append(f"Page: {page}")

                rows = [
                    [Paragraph("  |  ".join(meta_parts), self.s["ref_header"])],
                    [Paragraph(excerpt, self.s["ref_body"])],
                ]
                t = Table(rows, colWidths=[CONTENT_W])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (0, 0), LIGHT_TEAL),
                    ("BACKGROUND",    (0, 1), (0, 1), WHITE),
                    ("BOX",           (0, 0), (-1, -1), 0.6, TEAL),
                    ("TOPPADDING",    (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ]))
                out.append(t)
                out.append(self.spacer(2))
            elif not _is_empty(chunk):
                # plain string chunk — truncate
                text = str(chunk)
                if len(text) > self.MAX_EXCERPT:
                    text = text[:self.MAX_EXCERPT].rsplit(" ", 1)[0] + " …"
                out.append(Paragraph(f"[{i}]  {text}", self.s["bullet"]))

        return out

    # ── Header ────────────────────────────────────────────────────────────

    def format_header(self, report_type: str) -> List:
        out = [
            Paragraph("CARDIOAI", self.s["brand"]),
            self.spacer(3),
            Paragraph(
                "AI-Powered Multi-Modal Heart Disease Prediction System",
                self.s["brand_sub"],
            ),
            self.spacer(2),
            Paragraph(
                "Doctor Decision Support &amp; Patient Risk Assessment",
                self.s["tagline"],
            ),
            self.spacer(1),
            HRFlowable(width="100%", thickness=2, color=TEAL,
                       spaceAfter=5, spaceBefore=2),
        ]
        badge = Table(
            [[Paragraph(f"{report_type} Report", self.s["badge"])]],
            colWidths=[CONTENT_W],
        )
        badge.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        out.append(badge)
        return out

    # ── Footer ────────────────────────────────────────────────────────────

    def format_footer(self) -> List:
        return [
            self.spacer(6),
            HRFlowable(width="100%", thickness=1, color=TEAL,
                       spaceAfter=4, spaceBefore=2),
            Paragraph(
                f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}"
                "&#160; | &#160;Generated by CardioAI v2.0"
                "&#160; | &#160;Confidential Medical Document",
                self.s["footer"],
            ),
        ]

    # ── Generic section ───────────────────────────────────────────────────

    def section(self, title: str, data: Any) -> List:
        """Fallback: auto-dispatch based on type."""
        if _is_empty(data):
            return []
        out = [self.spacer(3), self.section_banner(title), self.spacer(2)]
        if isinstance(data, dict):
            out.extend(self.format_dict(data))
        elif isinstance(data, list):
            out.extend(self.format_list(data))
        else:
            out.append(Paragraph(str(data), self.s["plain"]))
        return out