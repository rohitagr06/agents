"""
tests/test_orchestrator.py — Pipeline orchestrator tests for MediScan AI.
Run standalone: uv run python tests/test_orchestrator.py
Run via pytest: uv run python -m pytest tests/test_orchestrator.py -v
"""

import asyncio
import os
import time
from unittest.mock import MagicMock
import pytest

from pipeline.orchestrator import (
    MediScanOrchestrator,
    SessionState,
    AnalysisResult,
    MAX_ANALYSES_PER_SESSION,
    COOLDOWN_SECONDS,
    _check_rate_limit,
    _compute_file_hash,
    _build_summary_md,
)
from output.pdf_builder import generate_pdf
from custom_data_types import (
    ReportFindings,
    ReportRecommendations,
    PatientContext,
    LabValue,
    AbnormalFlag,
    DietaryRecommendation,
    LifestyleModification,
    FollowUpAction,
)
from tools.report_analyzer import format_findings_for_display
from tools.recommendation_generator import format_recommendations_for_display


# ── Sample PDF path — used by LLM tests ──────────────────────

SAMPLE_PDF = None
for _path in ["tests/sample_report.pdf", "sample_report.pdf"]:
    if os.path.exists(_path):
        SAMPLE_PDF = _path
        break


# ── Shared mock data — used by multiple tests ─────────────────

MOCK_PARSED = MagicMock()
MOCK_PARSED.file_name = "test_report.pdf"
MOCK_PARSED.word_count = 1200
MOCK_PARSED.page_count = 4
MOCK_PARSED.chunk_count = 1

MOCK_FINDINGS = ReportFindings(
    report_type="lab_report",
    patient_context=PatientContext(
        patient_name="Test Patient",
        age="35 years",
        gender="Male",
        report_date="01/Jan/2025",
    ),
    lab_values=[
        LabValue(
            parameter="Hemoglobin",
            value="11.2 g/dL",
            reference_range="13.0-17.0",
            flag="Low",
            clinical_note="Below reference range",
        ),
        LabValue(
            parameter="Glucose",
            value="118 mg/dL",
            reference_range="70-99 mg/dL",
            flag="High",
        ),
    ],
    abnormal_flags=[
        AbnormalFlag(
            finding="Hemoglobin low at 11.2 g/dL",
            severity="moderate",
            category="hematology",
        ),
        AbnormalFlag(
            finding="Glucose elevated at 118 mg/dL",
            severity="mild",
            category="metabolic",
        ),
    ],
    clinical_summary=(
        "This lab report for a 35-year-old male shows hemoglobin below "
        "the normal range at 11.2 g/dL, and fasting glucose in the "
        "pre-diabetic range at 118 mg/dL."
    ),
    confidence="high",
)

MOCK_RECS = ReportRecommendations(
    overall_urgency="consult_soon",
    overall_assessment=(
        "Your results show two areas needing attention: mild anemia and "
        "borderline glucose. Both are manageable with lifestyle changes "
        "and a follow-up with your doctor within the next few weeks."
    ),
    dietary_recommendations=[
        DietaryRecommendation(
            suggestion="Increase iron-rich foods",
            reason="Hemoglobin is below the normal range",
            priority="high",
            foods_to_increase=["spinach", "lentils", "lean red meat"],
            foods_to_avoid=["tea with meals"],
        ),
    ],
    lifestyle_modifications=[
        LifestyleModification(
            modification="30 minutes of brisk walking 5 days/week",
            reason="Regular aerobic exercise helps lower fasting glucose",
            category="exercise",
            priority="medium",
        ),
    ],
    follow_up_actions=[
        FollowUpAction(
            action="Repeat CBC in 6 weeks to monitor hemoglobin",
            timeframe="Within 6 weeks",
            urgency="soon",
            specialist=None,
        ),
    ],
)


# ── Unit tests — no API calls ─────────────────────────────────

def test_session_state_defaults():
    """SessionState initialises with correct defaults."""
    state = SessionState()
    assert state.analyses_used == 0
    assert state.last_analysis_time == 0.0
    assert state.cache == {}


def test_rate_limit_fresh_session():
    """Fresh session passes rate limit check."""
    state = SessionState()
    allowed, msg = _check_rate_limit(state)
    assert allowed, f"Fresh session should be allowed, got: {msg}"


def test_rate_limit_cooldown_active():
    """Rate limit blocks during cooldown period."""
    state = SessionState(last_analysis_time=time.time())
    allowed, msg = _check_rate_limit(state)
    assert not allowed
    assert "wait" in msg.lower() or "seconds" in msg.lower()


def test_rate_limit_cooldown_expired():
    """Rate limit allows after cooldown expires."""
    state = SessionState(
        last_analysis_time=time.time() - (COOLDOWN_SECONDS + 1)
    )
    allowed, msg = _check_rate_limit(state)
    assert allowed, f"Should be allowed after cooldown, got: {msg}"


def test_rate_limit_session_cap():
    """Rate limit blocks when session cap is reached."""
    state = SessionState(analyses_used=MAX_ANALYSES_PER_SESSION)
    allowed, msg = _check_rate_limit(state)
    assert not allowed
    assert "session" in msg.lower()


def test_rate_limit_constants():
    """Rate limit constants match expected values."""
    assert MAX_ANALYSES_PER_SESSION == 2
    assert COOLDOWN_SECONDS == 60


def test_analysis_result_defaults():
    """AnalysisResult default fields are all correct."""
    result = AnalysisResult()
    assert result.success
    assert result.findings_md == ""
    assert result.recommendations_md == ""
    assert result.summary_md == ""
    assert result.raw_md == ""
    assert result.status == ""
    assert result.elapsed_seconds == 0.0
    assert not result.from_cache
    assert result.findings is None
    assert result.recommendations is None


def test_analysis_result_error_construction():
    """AnalysisResult error fields set correctly."""
    result = AnalysisResult(success=False, status="Test error", error="Test error")
    assert not result.success
    assert result.error == "Test error"


def test_build_summary_md_sections():
    """_build_summary_md() produces all required sections."""
    summary = _build_summary_md(MOCK_PARSED, MOCK_FINDINGS, MOCK_RECS)

    assert (
        "Clinical Overview" in summary
        or "Key Findings" in summary
        or "Findings" in summary
    ), "Section 1 (Clinical Overview) missing"
    assert (
        "Assessment" in summary
        or "Recommendations" in summary
        or "Advisor" in summary
    ), "Section 2 (Assessment) missing"
    assert (
        "Urgency" in summary
        or "Next Steps" in summary
        or "Glance" in summary
    ), "Section 3 (Urgency) missing"
    assert "11.2" in summary or "Hemoglobin" in summary
    assert "consult" in summary.lower() or "soon" in summary.lower()
    assert len(summary) > 300


def test_format_findings_markdown():
    """format_findings_for_display() produces well-formed markdown."""
    md = format_findings_for_display(MOCK_FINDINGS)
    assert "## " in md or "# " in md
    assert "Hemoglobin" in md
    assert "Low" in md or "High" in md
    assert len(md) > 200


def test_format_recommendations_markdown():
    """format_recommendations_for_display() produces well-formed markdown."""
    md = format_recommendations_for_display(MOCK_RECS)
    assert "## " in md or "# " in md
    assert "consult" in md.lower() or "soon" in md.lower()
    assert len(md) > 200


def test_format_findings_non_medical():
    """format_findings_for_display() handles non-medical documents."""
    non_medical = ReportFindings(
        report_type="unknown",
        patient_context=PatientContext(),
        lab_values=[],
        abnormal_flags=[],
        clinical_summary="Not a medical report.",
        is_non_medical=True,
        confidence="low",
    )
    md = format_findings_for_display(non_medical)
    assert len(md) > 0


async def test_parse_failure():
    """Orchestrator handles parse failure gracefully."""
    orchestrator = MediScanOrchestrator()
    state = SessionState()

    result = None
    async for update in orchestrator.run(None, state):
        if isinstance(update, AnalysisResult):
            result = update

    assert result is not None
    assert not result.success
    assert result.error != ""
    assert result.status != ""
    assert result.findings_md == ""
    assert result.recommendations_md == ""


async def test_rate_limit():
    """Orchestrator blocks correctly when rate limit is hit."""
    orchestrator = MediScanOrchestrator()

    # Session cap reached
    state = SessionState(
        analyses_used=MAX_ANALYSES_PER_SESSION,
        last_analysis_time=time.time() - (COOLDOWN_SECONDS + 5),
    )
    result = None
    async for update in orchestrator.run(None, state):
        if isinstance(update, AnalysisResult):
            result = update

    assert result is not None
    assert not result.success
    assert (
        "session" in result.error.lower()
        or "analyses" in result.error.lower()
    )

    # Cooldown active
    state2 = SessionState(analyses_used=0, last_analysis_time=time.time())
    result2 = None
    async for update in orchestrator.run(None, state2):
        if isinstance(update, AnalysisResult):
            result2 = update

    assert result2 is not None
    assert not result2.success
    assert (
        "wait" in result2.error.lower()
        or "seconds" in result2.error.lower()
    )


async def test_cache_hit():
    """Cache hit returns result instantly without LLM call."""
    if SAMPLE_PDF is None:
        pytest.skip("No sample PDF found at tests/sample_report.pdf")

    orchestrator = MediScanOrchestrator()
    mock_file = MagicMock()
    mock_file.name = os.path.abspath(SAMPLE_PDF)

    cached_result = AnalysisResult(
        success=True,
        findings_md="cached findings",
        recommendations_md="cached recs",
        summary_md="cached summary",
        raw_md="cached raw",
        status="cached status",
        findings=MOCK_FINDINGS,
        recommendations=MOCK_RECS,
    )

    file_hash = _compute_file_hash(mock_file)
    state = SessionState()
    if file_hash:
        state.cache[file_hash] = cached_result

    result = None
    t_start = time.time()
    async for update in orchestrator.run(mock_file, state):
        if isinstance(update, AnalysisResult):
            result = update
    t_elapsed = time.time() - t_start

    assert result is not None
    assert result.success
    assert result.from_cache
    assert "[Cached]" in result.status or "Cached" in result.status
    assert t_elapsed < 2.0, f"Cache hit too slow: {t_elapsed:.2f}s"


# ── LLM tests — real API calls ────────────────────────────────

@pytest.mark.llm
async def test_full_pipeline():
    """Full pipeline end-to-end with real PDF and real LLM calls."""
    if SAMPLE_PDF is None:
        pytest.skip("No sample PDF found at tests/sample_report.pdf")

    orchestrator = MediScanOrchestrator()
    state = SessionState()
    mock_file = MagicMock()
    mock_file.name = os.path.abspath(SAMPLE_PDF)

    status_messages = []
    result = None
    updated_state = state

    async for update in orchestrator.run(mock_file, state):
        if isinstance(update, str):
            status_messages.append(update)
        elif isinstance(update, SessionState):
            updated_state = update
        elif isinstance(update, AnalysisResult):
            result = update

    assert result is not None, "Pipeline yielded no AnalysisResult"
    assert result.success, f"Pipeline failed: {result.error}"
    assert len(result.findings_md) > 100
    assert len(result.recommendations_md) > 100
    assert len(result.summary_md) > 100
    assert len(result.raw_md) > 50
    assert result.elapsed_seconds > 0
    assert result.findings is not None
    assert result.recommendations is not None
    assert not result.from_cache
    assert updated_state.analyses_used == 1
    assert updated_state.last_analysis_time > 0
    assert any("Parsing" in m or "parsed" in m.lower() for m in status_messages)
    assert any("Analyz" in m for m in status_messages)
    assert any("Recommend" in m for m in status_messages)


@pytest.mark.llm
async def test_pdf_generation():
    """PDF generation produces a valid, correctly-sized PDF file."""
    pdf_path = generate_pdf(MOCK_FINDINGS, MOCK_RECS)

    assert os.path.exists(pdf_path), f"PDF not found at: {pdf_path}"

    size_kb = os.path.getsize(pdf_path) / 1024
    assert size_kb > 5, f"PDF too small: {size_kb:.1f} KB"
    assert size_kb < 5000, f"PDF too large: {size_kb:.1f} KB"

    with open(pdf_path, "rb") as f:
        header = f.read(4)
    assert header == b"%PDF", f"Not a valid PDF. Header: {header}"

    os.unlink(pdf_path)


# ── Standalone script mode ────────────────────────────────────

async def _main():
    print("\n" + "═" * 65)
    print("  MediScan AI — Orchestrator Test")
    print("═" * 65)

    print("\n[1/5] SessionState and rate limit logic...")
    test_session_state_defaults()
    test_rate_limit_fresh_session()
    test_rate_limit_cooldown_active()
    test_rate_limit_cooldown_expired()
    test_rate_limit_session_cap()
    test_rate_limit_constants()
    print("   ✅ All rate limit tests passed")

    print("\n[2/5] AnalysisResult dataclass...")
    test_analysis_result_defaults()
    test_analysis_result_error_construction()
    print("   ✅ AnalysisResult fields correct")

    print("\n[3/5] _build_summary_md()...")
    test_build_summary_md_sections()
    print("   ✅ Summary markdown has all required sections")

    print("\n[4/5] Parse failure and rate limit recovery...")
    await test_parse_failure()
    await test_rate_limit()
    print("   ✅ Graceful error handling confirmed")

    print("\n[5/5] PDF generation...")
    await test_pdf_generation()
    print("   ✅ PDF generated, validated, and cleaned up")

    if SAMPLE_PDF:
        print("\n[Bonus] Full pipeline with real PDF...")
        await test_full_pipeline()
        print("   ✅ End-to-end pipeline passed")
    else:
        print("\n⚠️  Skipping full pipeline — no sample PDF found")
        print("   Add: tests/sample_report.pdf")

    print("\n" + "═" * 65)
    print("  ✅ ALL TESTS PASSED")
    print("═" * 65)


if __name__ == "__main__":
    asyncio.run(_main())