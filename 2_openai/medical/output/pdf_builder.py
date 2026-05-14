"""
output/pdf_builder.py — Downloadable PDF Report Generator for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE:
    Generates a professional, downloadable PDF report from ReportFindings
    and ReportRecommendations using ReportLab.

    Called ONLY when the user clicks the "Download PDF" button in app.py.
    Output is a temp file — Gradio serves it for download and cleans it up.

STRUCTURE:
    Page 1   : Cover page (white bg + medical blue header bar at top)
    Page 2   : Patient context table + clinical summary
    Page 2+  : Lab values table (color-coded flag column)
    Page 2+  : Abnormal findings list (sorted by severity)
    Page 2+  : Medications table (if present)
    Page 3+  : Recommendations (urgency banner, dietary, lifestyle, follow-up)
    All pages: Blue header bar (except cover) + page number footer

DESIGN:
    — Helvetica throughout (built into ReportLab, no install needed)
    — Medical blue (#1B4F8A) headers, white background — clean and clinical
    — Color-coded flag column: green Normal, red High, orange Low, amber Borderline
    — KeepTogether() on every recommendation block — no orphaned headings
    — No raw extracted text in PDF (per spec)
    — Temp file only — deleted after download (per spec)

FIELD MAPPING (matches user's custom_data_types.py exactly):
    DietaryRecommendation : .suggestion, .reason, .priority, .foods_to_increase, .foods_to_avoid
    LifestyleModification : .modification, .reason, .priority
    FollowUpAction        : .action, .timeframe, .urgency, .specialist
    ReportRecommendations : .overall_urgency, .overall_assessment, .disclaimer
"""

import logging
import os
from datetime import datetime
import tempfile

from custom_data_types import ReportFindings, ReportRecommendations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("pdf_builder")

# ─────────────────────────────────────────────────────────────
#  Brand Colors
# ─────────────────────────────────────────────────────────────

MEDICAL_BLUE = colors.HexColor("#1B4F8A")
LIGHT_BLUE = colors.HexColor("#E8F0FB")
DARK_TEXT = colors.HexColor("#1A1A2E")
GREY_TEXT = colors.HexColor("#5A6A7A")
WHITE = colors.white

FLAG_COLORS = {
    "Normal": colors.HexColor("#27AE60"),
    "Low": colors.HexColor("#E67E22"),
    "High": colors.HexColor("#E74C3C"),
    "Borderline": colors.HexColor("#F39C12"),
    "Critical": colors.HexColor("#8E1010"),
    "Unknown": colors.HexColor("#7F8C8D"),
}

SEVERITY_COLORS = {
    "critical": colors.HexColor("#8E1010"),
    "severe": colors.HexColor("#E74C3C"),
    "moderate": colors.HexColor("#F39C12"),
    "mild": colors.HexColor("#27AE60"),
}

URGENCY_LABELS = {
    "routine": "✅  ROUTINE — Standard follow-up recommended",
    "consult_soon": "🟡  ATTENTION — Consult your doctor soon",
    "urgent": "🔴  URGENT — Seek medical attention promptly",
    "seek_immediate_care": "🚨  IMMEDIATE — Seek emergency care now",
}

URGENCY_BG_COLORS = {
    "routine": colors.HexColor("#1E8449"),
    "consult_soon": colors.HexColor("#B7770D"),
    "urgent": colors.HexColor("#922B21"),
    "seek_immediate_care": colors.HexColor("#6E0000"),
}

PRIORITY_COLORS = {
    "high": "#E74C3C",
    "medium": "#F39C12",
    "low": "#27AE60",
}

# ─────────────────────────────────────────────────────────────
#  Page Layout
# ─────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
HEADER_H = 14 * mm
FOOTER_H = 10 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# Shared table base style — applied to all tables, customized per table
_BASE_STYLE = [
    ("BACKGROUND", (0, 0), (-1, 0), MEDICAL_BLUE),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8D8EE")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]

# ─────────────────────────────────────────────────────────────
#  Style Factory
# ─────────────────────────────────────────────────────────────


def _build_styles() -> dict:
    return {
        # cover
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=MEDICAL_BLUE,
            alignment=TA_CENTER,
            spaceAfter=10,
            leading=32,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            fontName="Helvetica",
            fontSize=12,
            textColor=GREY_TEXT,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "cover_label": ParagraphStyle(
            "cover_label",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=GREY_TEXT,
            alignment=TA_CENTER,
            spaceAfter=1,
            leading=12,
        ),
        "cover_value": ParagraphStyle(
            "cover_value",
            fontName="Helvetica",
            fontSize=11,
            textColor=DARK_TEXT,
            alignment=TA_CENTER,
            spaceAfter=8,
            leading=16,
        ),
        "cover_disclaimer": ParagraphStyle(
            "cover_disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            textColor=GREY_TEXT,
            alignment=TA_CENTER,
            leading=11,
        ),
        "header_brand": ParagraphStyle(
            "header_brand",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        # Section & body
        "section_title": ParagraphStyle(
            "section_title",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=MEDICAL_BLUE,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=DARK_TEXT,
            leading=14,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Helvetica",
            fontSize=8,
            textColor=GREY_TEXT,
            leading=12,
            spaceAfter=2,
        ),
        "small_italic": ParagraphStyle(
            "small_italic",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=GREY_TEXT,
            leading=12,
            spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9,
            textColor=DARK_TEXT,
            leading=14,
            leftIndent=12,
            spaceAfter=3,
        ),
        # Table cells
        "th": ParagraphStyle(
            "th",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "td": ParagraphStyle(
            "td",
            fontName="Helvetica",
            fontSize=8.5,
            textColor=DARK_TEXT,
            leading=12,
        ),
        "td_c": ParagraphStyle(
            "td_c",
            fontName="Helvetica",
            fontSize=8.5,
            textColor=DARK_TEXT,
            alignment=TA_CENTER,
            leading=12,
        ),
        # Recommendations
        "rec_title": ParagraphStyle(
            "rec_title",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=DARK_TEXT,
            leading=14,
            spaceAfter=2,
        ),
        "rec_reason": ParagraphStyle(
            "rec_reason",
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            textColor=GREY_TEXT,
            leading=13,
            leftIndent=10,
            spaceAfter=4,
        ),
        "rec_foods": ParagraphStyle(
            "rec_foods",
            fontName="Helvetica",
            fontSize=8,
            textColor=DARK_TEXT,
            leading=12,
            leftIndent=10,
            spaceAfter=5,
        ),
        "urgency_text": ParagraphStyle(
            "urgency_text",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
    }


# ─────────────────────────────────────────────────────────────
#  Custom Doc Template — header bar + footer on content pages
# ─────────────────────────────────────────────────────────────


class _MediScanDoc(BaseDocTemplate):
    """
    Paints the blue header bar and page-number footer on every
    content page via afterPage(). Cover page (page 1) is skipped.
    """

    def __init__(self, path: str, patient_name: str, **kwargs):
        super().__init__(path, **kwargs)
        self._patient_name = patient_name

    def afterPage(self):
        canvas = self.canv
        page_num = canvas.getPageNumber()
        if page_num == 1:
            return  # cover page handles its own design

        canvas.saveState()

        # Blue header bar
        canvas.setFillColor(MEDICAL_BLUE)
        canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(WHITE)
        canvas.drawString(MARGIN, PAGE_H - HEADER_H + 4.5 * mm, "MediScan AI")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#BED3F0"))
        canvas.drawRightString(
            PAGE_W - MARGIN,
            PAGE_H - HEADER_H + 4.5 * mm,
            f"Medical Report  ·  {self._patient_name}",
        )

        # Light footer strip
        canvas.setFillColor(colors.HexColor("#F0F4FA"))
        canvas.rect(0, 0, PAGE_W, FOOTER_H, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#D0DCF0"))
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, FOOTER_H, PAGE_W - MARGIN, FOOTER_H)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY_TEXT)
        canvas.drawCentredString(
            PAGE_W / 2,
            3 * mm,
            f"Page {page_num}  ·  MediScan AI  ·  For informational purposes only",
        )

        canvas.restoreState()


def _make_doc(path: str, patient_name: str) -> _MediScanDoc:
    doc = _MediScanDoc(path, patient_name=patient_name, pagesize=A4)

    cover_frame = Frame(
        0,
        0,
        PAGE_W,
        PAGE_H,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="cover",
    )
    content_frame = Frame(
        MARGIN,
        FOOTER_H + 4 * mm,
        CONTENT_W,
        PAGE_H - HEADER_H - FOOTER_H - 8 * mm,
        leftPadding=0,
        rightPadding=0,
        topPadding=4 * mm,
        bottomPadding=4 * mm,
        id="content",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover_tpl", frames=[cover_frame]),
            PageTemplate(id="content_tpl", frames=[content_frame]),
        ]
    )
    return doc


# ─────────────────────────────────────────────────────────────
#  Cover Page
#
#  Two-row table fills the entire page:
#  Row 0 (blue, 18mm)  — "MediScan AI" branding
#  Row 1 (white, rest) — patient details, report type, disclaimer
# ─────────────────────────────────────────────────────────────


def _cover_page(
    s: dict, findings: ReportFindings, patient_name: str, generated_at: str
) -> list:
    ctx = findings.patient_context
    report_type = findings.report_type.replace("_", " ").title()

    header_cell = [Paragraph("MediScan AI", s["header_brand"])]

    body_cell = [
        Spacer(1, 22 * mm),
        Paragraph("Medical Report Analysis", s["cover_title"]),
        Spacer(1, 3 * mm),
        Paragraph("AI-Powered Health Insights", s["cover_subtitle"]),
        Spacer(1, 10 * mm),
        HRFlowable(width=50 * mm, thickness=1.5, color=MEDICAL_BLUE, hAlign="CENTER"),
        Spacer(1, 12 * mm),
        Paragraph("PATIENT", s["cover_label"]),
        Paragraph(patient_name, s["cover_value"]),
        Paragraph("REPORT TYPE", s["cover_label"]),
        Paragraph(report_type, s["cover_value"]),
        Paragraph("GENERATED ON", s["cover_label"]),
        Paragraph(generated_at, s["cover_value"]),
    ]

    if ctx.report_date:
        body_cell += [
            Paragraph("REPORT DATE", s["cover_label"]),
            Paragraph(ctx.report_date, s["cover_value"]),
        ]

    if ctx.ordering_physician:
        body_cell += [
            Paragraph("ORDERED BY", s["cover_label"]),
            Paragraph(ctx.ordering_physician, s["cover_value"]),
        ]

    body_cell += [
        Spacer(1, 12 * mm),
        HRFlowable(
            width=80 * mm,
            thickness=0.5,
            color=colors.HexColor("#C8D8EE"),
            hAlign="CENTER",
        ),
        Spacer(1, 8 * mm),
        Paragraph(
            "WARNING: This report is generated by an AI system for informational and "
            "educational purposes only. It does NOT constitute professional medical "
            "advice, diagnosis, or treatment. Always consult a qualified healthcare "
            "provider regarding your results.",
            s["cover_disclaimer"],
        ),
        Spacer(1, 6 * mm),
    ]

    cover = Table(
        [[header_cell], [body_cell]],
        colWidths=[PAGE_W],
        rowHeights=[18 * mm, PAGE_H - 18 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), MEDICAL_BLUE),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, 1), WHITE),
                ("VALIGN", (0, 1), (-1, 1), "TOP"),
                ("ALIGN", (0, 1), (-1, 1), "CENTER"),
                ("LEFTPADDING", (0, 1), (-1, 1), 30 * mm),
                ("RIGHTPADDING", (0, 1), (-1, 1), 30 * mm),
                ("LINEBELOW", (0, 0), (-1, 0), 2, colors.HexColor("#4A80C4")),
            ]
        )
    )

    return [cover, NextPageTemplate("content_tpl"), PageBreak()]


# ─────────────────────────────────────────────────────────────
#  Section Builders
# ─────────────────────────────────────────────────────────────


def _patient_section(s: dict, findings: ReportFindings) -> list:
    ctx = findings.patient_context
    rows = [
        [Paragraph("Field", s["th"]), Paragraph("Value", s["th"])],
        [
            Paragraph("Patient Name", s["td"]),
            Paragraph(ctx.patient_name or "Not Specified", s["td"]),
        ],
        [
            Paragraph("Report Type", s["td"]),
            Paragraph(findings.report_type.replace("_", " ").title(), s["td"]),
        ],
        [
            Paragraph("Patient Age", s["td"]),
            Paragraph(ctx.age or "Not Specified", s["td"]),
        ],
        [
            Paragraph("Gender", s["td"]),
            Paragraph(ctx.gender or "Not Specified", s["td"]),
        ],
        [
            Paragraph("Report Date", s["td"]),
            Paragraph(ctx.report_date or "Not Specified", s["td"]),
        ],
        [
            Paragraph("Ordering Physician", s["td"]),
            Paragraph(ctx.ordering_physician or "Not Specified", s["td"]),
        ],
        [
            Paragraph("AI Confidence", s["td"]),
            Paragraph(findings.confidence.title(), s["td"]),
        ],
    ]
    style = list(_BASE_STYLE)
    for i in range(1, len(rows)):
        style.append(
            ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE if i % 2 == 0 else WHITE)
        )

    t = Table(rows, colWidths=[55 * mm, CONTENT_W - 55 * mm])
    t.setStyle(TableStyle(style))
    return [t, Spacer(1, 5 * mm)]


def _lab_values_section(s: dict, findings: ReportFindings) -> list:
    if not findings.lab_values:
        return []

    col_w = [CONTENT_W * 0.33, CONTENT_W * 0.17, CONTENT_W * 0.32, CONTENT_W * 0.18]
    rows = [
        [
            Paragraph("Parameter", s["th"]),
            Paragraph("Result", s["th"]),
            Paragraph("Reference Range", s["th"]),
            Paragraph("Status", s["th"]),
        ]
    ]
    style = list(_BASE_STYLE)

    for i, lv in enumerate(findings.lab_values, 1):
        flag_color = FLAG_COLORS.get(lv.flag, colors.HexColor("#7F8C8D"))
        rows.append(
            [
                Paragraph(lv.parameter, s["td"]),
                Paragraph(lv.value, s["td_c"]),
                Paragraph(lv.reference_range, s["td"]),
                Paragraph(
                    f'<font color="{flag_color.hexval()}"><b>{lv.flag}</b></font>',
                    s["td_c"],
                ),
            ]
        )
        # Alternating row + flag cell background
        style.append(
            ("BACKGROUND", (0, i), (2, i), LIGHT_BLUE if i % 2 == 0 else WHITE)
        )
        if lv.flag in ("High", "Critical"):
            style.append(("BACKGROUND", (3, i), (3, i), colors.HexColor("#FFEBEE")))
        elif lv.flag in ("Low", "Borderline"):
            style.append(("BACKGROUND", (3, i), (3, i), colors.HexColor("#FFF8E1")))
        else:
            style.append(("BACKGROUND", (3, i), (3, i), colors.HexColor("#E8F5E9")))

    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle(style))
    return [t, Spacer(1, 5 * mm)]


def _abnormal_section(s: dict, findings: ReportFindings) -> list:
    if not findings.abnormal_flags:
        return []

    order = {"critical": 0, "severe": 1, "moderate": 2, "mild": 3}
    sorted_ = sorted(findings.abnormal_flags, key=lambda f: order.get(f.severity, 4))
    elements = [Paragraph("⚠️  Abnormal Findings", s["section_title"])]

    for flag in sorted_:
        c = SEVERITY_COLORS.get(flag.severity, GREY_TEXT)
        elements.append(
            Paragraph(
                f'• <font color="{c.hexval()}"><b>[{flag.severity.upper()}]</b></font>  {flag.finding}',
                s["bullet"],
            )
        )
    elements.append(Spacer(1, 4 * mm))
    return elements


def _medications_section(s: dict, findings: ReportFindings) -> list:
    if not findings.medications:
        return []

    col_w = [CONTENT_W * 0.35, CONTENT_W * 0.28, CONTENT_W * 0.37]
    rows = [
        [
            Paragraph("Medication", s["th"]),
            Paragraph("Dosage", s["th"]),
            Paragraph("Purpose", s["th"]),
        ]
    ]
    style = list(_BASE_STYLE)
    for i, med in enumerate(findings.medications, 1):
        rows.append(
            [
                Paragraph(med.name, s["td"]),
                Paragraph(med.dosage, s["td"]),
                Paragraph(med.purpose or "—", s["td"]),
            ]
        )
        style.append(
            ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE if i % 2 == 0 else WHITE)
        )

    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle(style))
    return [t, Spacer(1, 5 * mm)]


def _recommendations_section(s: dict, recs: ReportRecommendations) -> list:
    elements = [
        Paragraph("💡  Personalized Recommendations", s["section_title"]),
        HRFlowable(width=CONTENT_W, thickness=1, color=MEDICAL_BLUE, spaceAfter=6),
    ]

    # Urgency banner
    label = URGENCY_LABELS.get(recs.overall_urgency, recs.overall_urgency.upper())
    banner_bg = URGENCY_BG_COLORS.get(recs.overall_urgency, MEDICAL_BLUE)
    banner = Table(
        [[Paragraph(label, s["urgency_text"])]],
        colWidths=[CONTENT_W],
        rowHeights=[10 * mm],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements += [banner, Spacer(1, 4 * mm)]
    elements.append(Paragraph(recs.overall_assessment, s["body"]))
    elements.append(Spacer(1, 4 * mm))

    # Dietary recommendations — uses .suggestion, .reason, .priority, .foods_to_increase, .foods_to_avoid
    if recs.dietary_recommendations:
        elements.append(Paragraph("🥗  Dietary Recommendations", s["section_title"]))
        for i, diet in enumerate(recs.dietary_recommendations, 1):
            pc = PRIORITY_COLORS.get(diet.priority.lower(), "#1B4F8A")
            block = [
                Paragraph(
                    f'<font color="{pc}"><b>[{diet.priority.upper()}]</b></font>  {i}. {diet.suggestion}',
                    s["rec_title"],
                ),
                Paragraph(f"Why: {diet.reason}", s["rec_reason"]),
            ]
            if diet.foods_to_increase:
                block.append(
                    Paragraph(
                        f"✅  Increase: {', '.join(diet.foods_to_increase)}",
                        s["rec_foods"],
                    )
                )
            if diet.foods_to_avoid:
                block.append(
                    Paragraph(
                        f"❌  Reduce/Avoid: {', '.join(diet.foods_to_avoid)}",
                        s["rec_foods"],
                    )
                )
            block.append(Spacer(1, 2 * mm))
            elements.append(KeepTogether(block))

    # Lifestyle modifications — uses .modification, .reason, .priority
    if recs.lifestyle_modifications:
        elements.append(Paragraph("🏃  Lifestyle Modifications", s["section_title"]))
        for i, ls in enumerate(recs.lifestyle_modifications, 1):
            block = [
                Paragraph(f"{i}. {ls.modification}", s["rec_title"]),
                Paragraph(f"Why: {ls.reason}", s["rec_reason"]),
                Spacer(1, 2 * mm),
            ]
            elements.append(KeepTogether(block))

    # Follow-up actions — uses .action, .timeframe, .urgency, .specialist
    if recs.follow_up_actions:
        elements.append(Paragraph("📅  Follow-Up Actions", s["section_title"]))
        for i, fa in enumerate(recs.follow_up_actions, 1):
            specialist_str = f"  →  {fa.specialist}" if fa.specialist else ""
            urgency_str = f"  |  Priority: {fa.urgency.title()}" if fa.urgency else ""
            block = [
                Paragraph(f"{i}. {fa.action}", s["rec_title"]),
                Paragraph(
                    f"When: {fa.timeframe}{specialist_str}{urgency_str}",
                    s["small"],
                ),
                Spacer(1, 2 * mm),
            ]
            elements.append(KeepTogether(block))

    # Disclaimer
    elements += [
        Spacer(1, 6 * mm),
        HRFlowable(width=CONTENT_W, thickness=0.5, color=colors.HexColor("#C8D8EE")),
        Spacer(1, 3 * mm),
        Paragraph(recs.disclaimer, s["small_italic"]),
    ]
    return elements


# ─────────────────────────────────────────────────────────────
#  generate_pdf()  — PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────


def generate_pdf(
    findings: ReportFindings,
    recommendations: ReportRecommendations,
) -> str:
    """
    Generate a downloadable PDF report and return the temp file path.

    Args:
        findings        : ReportFindings from report_analyzer_agent
        recommendations : ReportRecommendations from recommendation_agent

    Returns:
        Absolute path to the generated temp .pdf file.
        Gradio's gr.File serves and cleans it up automatically.

    Raises:
        RuntimeError: Caught and logged in app.py.
    """
    ctx = findings.patient_context

    # Use extracted patient name if available, fall back to age · gender
    if ctx.patient_name:
        patient_name = ctx.patient_name
    else:
        name_parts = [p for p in [ctx.age, ctx.gender] if p]
        patient_name = "  ·  ".join(name_parts) if name_parts else "Not Specified"

    date_str = datetime.now().strftime("%Y-%m-%d")
    generated_at = datetime.now().strftime("%B %d, %Y  %H:%M")
    safe_seg = (
        patient_name.replace(" ", "_")
        .replace("·", "")
        .replace("/", "-")
        .strip("_")[:20]
        if patient_name != "Not Specified"
        else "Report"
    )
    filename = f"MediScan_{safe_seg}_{date_str}.pdf"

    # Temp file — Gradio streams this to the browser for download
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", prefix="mediscan_", delete=False)
    tmp_path = tmp.name
    tmp.close()

    logger.info(f"Generating PDF: {filename} → {tmp_path}")

    try:
        s = _build_styles()
        doc = _make_doc(tmp_path, patient_name)

        story = [NextPageTemplate("cover_tpl")]
        story += _cover_page(s, findings, patient_name, generated_at)

        # Findings pages
        story.append(Paragraph("🔬  Report Findings", s["section_title"]))
        story.append(
            HRFlowable(width=CONTENT_W, thickness=1, color=MEDICAL_BLUE, spaceAfter=6)
        )
        story.append(Paragraph(findings.clinical_summary, s["body"]))
        story.append(Spacer(1, 4 * mm))
        story += _patient_section(s, findings)
        story += _lab_values_section(s, findings)
        story += _abnormal_section(s, findings)
        story += _medications_section(s, findings)
        story.append(PageBreak())

        # Recommendations page
        story += _recommendations_section(s, recommendations)

        doc.build(story)

        kb = os.path.getsize(tmp_path) / 1024
        logger.info(f"PDF ready: {filename} ({kb:.1f} KB)")
        return tmp_path

    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        logger.error(f"PDF generation failed: {e}")
        raise RuntimeError(f"PDF generation failed: {e}") from e
