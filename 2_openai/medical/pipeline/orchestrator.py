"""
agents/orchestrator.py — Pipeline Orchestrator for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE:
    This is the central coordinator for the entire MediScan AI pipeline.
    It follows the exact same pattern as your Deep Research project's
    ResearchManager class — a clean async manager that:
    1. Runs each step in sequence
    2. Yields status updates so the Gradio UI can show progress
    3. Returns the final structured result

YOUR DEEP RESEARCH PATTERN (ResearchManager):
    async def run(self, query: str):
        yield "Searches planned, starting to search..."
        search_plan = await self.plan_searches(query)
        yield "Searches complete, writing report..."
        report = await self.write_report(query, search_results)
        yield report.markdown_report

MEDISCAN ORCHESTRATOR (same pattern):
    async def run(self, file):
        yield "📄 Parsing document..."
        parsed = await self.parse(file)
        yield "🔬 Analyzing findings..."
        findings = await self.analyze(parsed)
        yield "💡 Generating recommendations..."
        recommendations = await self.recommend(findings)
        yield "📋 Building summary..."
        summary = await self.summarize(findings, recommendations)
        yield result  ← AnalysisResult dataclass

WHY A SEPARATE ORCHESTRATOR FILE?
    app.py should only handle UI concerns — layout, events, rendering.
    All pipeline logic lives here. This means:
    — app.py stays thin and readable
    — The pipeline can be tested independently of Gradio
    — Future RC2 features (OCR, conversational follow-up) are added
      here without touching the UI layer
    — The same orchestrator could power a CLI or API endpoint

GRADIO ASYNC STREAMING PATTERN:
    Gradio 4.x supports async generator functions natively.
    When analyze_report() in app.py uses `yield`, Gradio streams
    each yielded value to the UI in real time. This is how we show
    "Parsing document..." → "Analyzing findings..." → final result
    without the UI freezing during the 20-30 second analysis.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from tools.document_parser import (
    parse_document,
    ParsedDocument,
    format_parsed_for_display,
)
from tools.report_analyzer import analyze_report_text, format_findings_for_display
from tools.recommendation_generator import (
    generate_recommendations,
    format_recommendations_for_display,
)
from custom_data_types import ReportFindings, ReportRecommendations
import config

logger = logging.getLogger("orchestrator")

# ─────────────────────────────────────────────────────────────
#  Rate Limiter Constants
# ─────────────────────────────────────────────────────────────

MAX_ANALYSES_PER_SESSION: int = 2  # per your answer: 2 per session
COOLDOWN_SECONDS: int = 60  # 60s cooldown between analyses

# ─────────────────────────────────────────────────────────────
#  SessionState — per-session rate limit + cache
#
#  Stored in gr.State() in app.py — one instance per browser session.
#  Cleared on page refresh (session-only, as per your answer).
#
#  WHY dataclass AND NOT A PLAIN DICT?
#  gr.State() can hold any Python object. A dataclass gives named
#  fields with type hints — much safer than dict["analyses_used"]
#  which silently returns KeyError if the key is missing.
#
#  Cache stores: file_hash → AnalysisResult
#  Key: MD5 of file bytes — same content = same hash regardless of filename
#  Scope: session-only — cleared on page refresh (privacy-safe for public app)
# ─────────────────────────────────────────────────────────────


@dataclass
class SessionState:
    """Per-session state for rate limiting and result caching."""

    analyses_used: int = 0
    last_analysis_time: float = 0.0
    cache: dict = field(default_factory=dict)  # hash → AnalysisResult


def _compute_file_hash(file) -> str:
    """
    Compute MD5 hash of file bytes for cache key.
    Returns empty string on any error — cache miss is safe.
    """
    try:
        path = file.name if hasattr(file, "name") else str(file)
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def _check_rate_limit(state: SessionState) -> tuple[bool, str]:
    """
    Check whether this session can run another analysis.

    Returns:
        (allowed: bool, message: str)
        If not allowed, message contains the reason + countdown.
    """
    now = time.time()

    # ── Cooldown check ────────────────────────────────────────
    if state.last_analysis_time > 0:
        elapsed = now - state.last_analysis_time
        remaining = int(COOLDOWN_SECONDS - elapsed)
        if remaining > 0:
            return False, (
                f"⏳ Please wait {remaining} seconds before the next analysis. "
                f"This keeps the service available for everyone on the free tier."
            )

    # ── Session cap check ─────────────────────────────────────
    if state.analyses_used >= MAX_ANALYSES_PER_SESSION:
        return False, (
            f"🚫 You have used all {MAX_ANALYSES_PER_SESSION} analyses for this session. "
            f"Please refresh the page to start a new session."
        )
    return True, ""


# ─────────────────────────────────────────────────────────────
#  AnalysisResult — the complete output of the pipeline
#
#  WHY A DATACLASS INSTEAD OF RETURNING MULTIPLE VALUES?
#  The orchestrator produces 5 pieces of output for the UI:
#  findings_md, recommendations_md, summary_md, raw_md, status.
#  Returning them as a tuple (str, str, str, str, str) is error-prone
#  — the caller must remember the order. A named dataclass makes
#  each field self-documenting and impossible to mix up.
#
#  The orchestrator builds this progressively — fields are populated
#  one by one as each pipeline step completes.
# ─────────────────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """
    The complete output of the MediScan AI analysis pipeline.
    All fields are pre-formatted markdown strings ready for Gradio display.
    """

    findings_md: str = ""
    recommendations_md: str = ""
    summary_md: str = ""
    raw_md: str = ""
    status: str = ""
    pdf_path: str = ""
    success: bool = True
    error: str = ""
    elapsed_seconds: float = 0.0  # processing time shown in status
    from_cache: bool = False  # True if served from session cache
    # Raw Pydantic objects — needed by pdf_builder, never sent to Gradio
    findings: Optional[ReportFindings] = None
    recommendations: Optional[ReportRecommendations] = None


# ─────────────────────────────────────────────────────────────
#  _build_summary_md() — executive summary builder
#
#  WHY NO SEPARATE LLM CALL FOR THE SUMMARY?
#  We considered adding a "writer agent" (like your Deep Research
#  writer_agent) to synthesize findings + recommendations into a
#  narrative summary. But for RC1, the analyzer already produces
#  a clinical_summary and the recommendation agent produces an
#  overall_assessment. Combining these in Python (no LLM call)
#  produces a rich, accurate summary with zero extra latency
#  and zero extra API cost.
#
#  RC2 could add a dedicated summarizer agent if needed.
# ─────────────────────────────────────────────────────────────


def _build_summary_md(
    parsed: ParsedDocument,
    findings: ReportFindings,
    recommendations: ReportRecommendations,
) -> str:
    """
    Build the executive summary markdown from pipeline outputs.
    No LLM call — synthesizes existing outputs in Python.

    Args:
        parsed:          ParsedDocument from document_parser
        findings:        ReportFindings from report_analyzer
        recommendations: ReportRecommendations from recommendation_generator

    Returns:
        Markdown string for the Summary tab in Gradio.
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────
    lines += [
        "## 📋 Executive Summary",
        "",
    ]

    # ── Patient & report info ─────────────────────────────────
    ctx = findings.patient_context
    patient_parts = []
    if ctx.age:
        patient_parts.append(f"Age: **{ctx.age}**")
    if ctx.gender:
        patient_parts.append(f"Gender: **{ctx.gender}**")
    if ctx.report_date:
        patient_parts.append(f"Report Date: **{ctx.report_date}**")

    if patient_parts:
        lines += [" &nbsp;·&nbsp; ".join(patient_parts), ""]

    lines += [
        f"**File:** `{parsed.file_name}` &nbsp;·&nbsp; "
        f"**Pages:** {parsed.page_count} &nbsp;·&nbsp; "
        f"**Type:** {findings.report_type.replace('_', ' ').title()} &nbsp;·&nbsp; "
        f"**Confidence:** {findings.confidence.title()}",
        "",
        "---",
        "",
    ]

    # ── Clinical summary from analyzer ────────────────────────
    if findings.clinical_summary:
        lines += [
            "### 🩺 Clinical Overview",
            "",
            findings.clinical_summary,
            "",
        ]

    # ── Overall assessment from recommendation agent ──────────
    if recommendations.overall_assessment:
        lines += [
            "### 💡 Health Advisor's Assessment",
            "",
            recommendations.overall_assessment,
            "",
        ]

    # ── Quick stats dashboard ─────────────────────────────────
    total = len(findings.lab_values)
    abnormal = len([lv for lv in findings.lab_values if lv.flag != "Normal"])
    normal = total - abnormal
    flags = len(findings.abnormal_flags)
    diet_count = len(recommendations.dietary_recommendations)
    lifestyle_count = len(recommendations.lifestyle_modifications)
    followup_count = len(recommendations.follow_up_actions)

    lines += [
        "### 📊 At a Glance",
        "",
        "| Category | Count |",
        "|---|---|",
        f"| Lab values tested | {total} |",
        f"| Within normal range | ✅ {normal} |",
        f"| Abnormal / borderline | {'🔴' if abnormal > 0 else '✅'} {abnormal} |",
        f"| Areas needing attention | {'⚠️' if flags > 0 else '✅'} {flags} |",
        f"| Dietary recommendations | 🥗 {diet_count} |",
        f"| Lifestyle modifications | 🏃 {lifestyle_count} |",
        f"| Follow-up actions | 📅 {followup_count} |",
        "",
    ]

    # ── Urgency summary ───────────────────────────────────────
    urgency_map = {
        "routine": (
            "✅",
            "Routine",
            "No immediate action needed. Review at next scheduled check-up.",
        ),
        "consult_soon": ("🟡", "Consult Soon", "See your physician within 2-4 weeks."),
        "urgent": ("🟠", "Urgent", "See your physician within 1 week."),
        "seek_immediate_care": (
            "🚨",
            "Seek Immediate Care",
            "Contact your doctor today.",
        ),
    }
    icon, label, desc = urgency_map.get(
        recommendations.overall_urgency, ("⚠️", recommendations.overall_urgency, "")
    )
    lines += [
        f"### {icon} Overall Urgency: {label}",
        "",
        desc,
        "",
        "---",
        "",
    ]

    # ── Top abnormal findings ─────────────────────────────────
    if findings.abnormal_flags:
        lines += ["### ⚠️ Key Findings Requiring Attention", ""]
        severity_order = {"critical": 0, "severe": 1, "moderate": 2, "mild": 3}
        sorted_flags = sorted(
            findings.abnormal_flags, key=lambda f: severity_order.get(f.severity, 4)
        )
        for flag in sorted_flags[:5]:  # top 5 max
            severity_icons = {
                "critical": "🚨",
                "severe": "🔴",
                "moderate": "🟠",
                "mild": "🟡",
            }
            icon = severity_icons.get(flag.severity, "⚠️")
            lines.append(f"- {icon} **{flag.severity.title()}:** {flag.finding}")
        lines.append("")

    # ── Footer / disclaimer ───────────────────────────────────
    lines += [
        "---",
        f"*Generated by MediScan AI {config.APP_VERSION} · "
        f"openai/gpt-4.1-mini via AI Models · "
        f"{parsed.word_count:,} words extracted from {parsed.page_count} page(s)*",
        "",
        f"*⚠️ {config.MEDICAL_DISCLAIMER}*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  MediScanOrchestrator — the main pipeline manager
#
#  — Methods for each step (parse, analyze, recommend)
#  — An async run() generator that yields status strings
#  — Returns the final AnalysisResult at the end
# ─────────────────────────────────────────────────────────────


class MediScanOrchestrator:
    """
    Orchestrates the full MediScan AI analysis pipeline.

    Usage in app.py:
        orchestrator = MediScanOrchestrator()
        async for update in orchestrator.run(file):
            if isinstance(update, str):
                status = update      # show in status bar
            elif isinstance(update, AnalysisResult):
                result = update      # populate all tabs
    """

    async def parse(self, file) -> ParsedDocument:
        """Step 1: Validate and parse the uploaded document."""
        logger.info("Orchestrator: parsing document...")
        return parse_document(file)

    async def analyze(self, parsed: ParsedDocument) -> ReportFindings:
        """Step 2: Extract structured findings using the analyzer agent."""
        logger.info("Orchestrator: analyzing report...")
        return await analyze_report_text(
            text=parsed.text,
            file_name=parsed.file_name,
            page_count=parsed.page_count,
        )

    async def recommend(self, findings: ReportFindings) -> ReportRecommendations:
        """Step 3: Generate personalized recommendations."""
        logger.info("Orchestrator: generating recommendations...")
        return await generate_recommendations(findings)

    async def run(self, file, session_state: SessionState):
        """
        Run the complete pipeline, yielding status strings then
        the final AnalysisResult.

        Args:
            file          : Gradio file object
            session_state : SessionState from gr.State() — rate limit + cache

        Yields:
            str          — status update messages for the UI progress bar
            SessionState   — updated session state (app.py saves this back)
            AnalysisResult — the complete final result (last yield)

        Usage in app.py:
            async for update in orchestrator.run(file, state):
                if isinstance(update, SessionState):
                    state = update
                elif isinstance(update, str):
                    current_status = update
                elif isinstance(update, AnalysisResult):
                    result = update
                ...
        """
        start_time = time.time()

        # ── Rate limit check ──────────────────────────────────
        # Enforced before any work is done — free tier protection.
        # 2 analyses per session, 60s cooldown between each.

        allowed, reason = _check_rate_limit(session_state)
        if not allowed:
            yield AnalysisResult(success=False, status=reason, error=reason)
            return

        # ── Cache check ───────────────────────────────────────
        # MD5 hash of file bytes — same file content = same hash.
        # Cache is session-only: cleared on page refresh.
        file_hash = _compute_file_hash(file)
        if file_hash and file_hash in session_state.cache:
            cached = session_state.cache[file_hash]
            cached.from_cache = True
            cached.status = f"⚡ [Cached] {cached.status}"
            logger.info(f"Cache hit: {file_hash[:8]}")
            yield cached
            return

        # ── Step 1: Parse ─────────────────────────────────────
        yield "📄 Step 1/4 — Parsing document..."
        parsed = await self.parse(file)

        if not parsed.success:
            yield AnalysisResult(status=parsed.error, success=False, error=parsed.error)
            return

        raw_md = format_parsed_for_display(parsed)

        if not parsed.is_meaningful:
            warning = parsed.warning or "⚠️ Document text too short to analyze."
            yield AnalysisResult(
                raw_md=raw_md,
                status=warning,
                success=False,
                error=warning,
            )
            return

        yield f"📄 Parsed: {parsed.file_name} ({parsed.word_count:,} words · {parsed.page_count} pages)"

        # ── Step 2: Analyze ───────────────────────────────────
        yield "🔬 Step 2/4 — Analyzing findings with AI..."

        findings = await self.analyze(parsed)
        findings_md = format_findings_for_display(findings)

        abnormal_count = len(findings.abnormal_flags)
        lab_count = len(findings.lab_values)
        yield f"🔬 Analysis complete: {lab_count} lab values · {abnormal_count} abnormal findings"

        # ── Step 3: Recommend ─────────────────────────────────
        yield "💡 Step 3/4 — Generating personalized recommendations..."

        recommendations = await self.recommend(findings)
        recommendations_md = format_recommendations_for_display(recommendations)

        diet_count = len(recommendations.dietary_recommendations)
        lifestyle_count = len(recommendations.lifestyle_modifications)
        yield f"💡 Recommendations ready: {diet_count} dietary · {lifestyle_count} lifestyle"

        # ── Step 4: Summarize ─────────────────────────────────
        yield "📋 Step 4/4 — Building executive summary..."

        summary_md = _build_summary_md(parsed, findings, recommendations)

        # ── Final status ──────────────────────────────────────
        elapsed = time.time() - start_time
        urgency_labels = {
            "routine": "✅ Routine",
            "consult_soon": "🟡 Consult Soon",
            "urgent": "🟠 Urgent",
            "seek_immediate_care": "🚨 Seek Immediate Care",
        }
        urgency_display = urgency_labels.get(
            recommendations.overall_urgency, recommendations.overall_urgency
        )

        remaining_analyses = MAX_ANALYSES_PER_SESSION - (
            session_state.analyses_used + 1
        )

        final_status = (
            f"✅ Complete — {parsed.file_name} · "
            f"{lab_count} values · {abnormal_count} flags · "
            f"Urgency: {urgency_display} · {elapsed:.1f}s · "
            f"{remaining_analyses} analysis remaining this session"
        )

        logger.info(f"Orchestrator: pipeline complete | {final_status}")

        # ── Update session state ──────────────────────────────
        session_state.analyses_used += 1
        session_state.last_analysis_time = time.time()

        # ── Yield final result ────────────────────────────────
        result = AnalysisResult(
            findings_md=findings_md,
            recommendations_md=recommendations_md,
            summary_md=summary_md,
            raw_md=raw_md,
            status=final_status,
            success=True,
            elapsed_seconds=elapsed,
            from_cache=False,
            findings=findings,  # stored for PDF download
            recommendations=recommendations,  # stored for PDF download
        )  # Store in session cache — same file won't re-run LLM
        if file_hash:
            session_state.cache[file_hash] = result

        # Yield updated state BEFORE result so app.py saves it first
        yield session_state
        yield result
