"""
tests/test_end_to_end.py — End-to-end QA test suite for MediScan AI.
Run standalone: uv run python tests/test_end_to_end.py
Run via pytest: uv run python -m pytest tests/test_end_to_end.py -v
"""

import asyncio
import os
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from utils.validator import validate_file
from utils.sanitizer import sanitize, is_meaningful, chunk_text, get_text_stats
from tools.document_parser import parse_document, format_parsed_for_display
from tools.report_analyzer import analyze_report_text, format_findings_for_display
from tools.recommendation_generator import (
    generate_recommendations,
    format_recommendations_for_display,
)
from pipeline.orchestrator import (
    MediScanOrchestrator,
    SessionState,
    AnalysisResult,
    MAX_ANALYSES_PER_SESSION,
    COOLDOWN_SECONDS,
    _compute_file_hash,
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
import config


# ── Test asset discovery ──────────────────────────────────────

SAMPLE_PDF = None
SAMPLE_DOCX = None

for _p in ["tests/sample_report.pdf", "sample_report.pdf"]:
    if os.path.exists(_p):
        SAMPLE_PDF = os.path.abspath(_p)
        break

for _p in ["tests/sample_docx.docx", "sample_docx.docx"]:
    if os.path.exists(_p):
        SAMPLE_DOCX = os.path.abspath(_p)
        break


# ── Helpers ───────────────────────────────────────────────────

def _mock_file(path: str) -> MagicMock:
    m = MagicMock()
    m.name = path
    return m


# ── Shared mock data ──────────────────────────────────────────

MOCK_FINDINGS = ReportFindings(
    report_type="lab_report",
    patient_context=PatientContext(
        age="33 years",
        gender="Male",
        report_date="13/Feb/2025",
        ordering_physician="Dr. Smith",
    ),
    lab_values=[
        LabValue(
            parameter="Hemoglobin",
            value="15.3 g/dL",
            reference_range="13.0-17.0",
            flag="Normal",
        ),
        LabValue(
            parameter="Uric Acid",
            value="8.5 mg/dL",
            reference_range="3.5-7.2",
            flag="High",
            clinical_note="Elevated — associated with gout risk",
        ),
        LabValue(
            parameter="LDL Cholesterol",
            value="131 mg/dL",
            reference_range="<100 desirable",
            flag="Borderline",
            clinical_note="Borderline high — cardiovascular risk factor",
        ),
    ],
    abnormal_flags=[
        AbnormalFlag(
            finding="Uric Acid elevated at 8.5 mg/dL",
            severity="moderate",
            category="metabolic",
        ),
        AbnormalFlag(
            finding="LDL Cholesterol borderline high at 131 mg/dL",
            severity="mild",
            category="lipid",
        ),
    ],
    clinical_summary=(
        "Lab report for a 33-year-old male. Uric acid is elevated "
        "and LDL cholesterol is borderline high."
    ),
    confidence="high",
)

MOCK_RECS = ReportRecommendations(
    overall_urgency="consult_soon",
    overall_assessment=(
        "Two areas need attention: elevated uric acid and borderline LDL. "
        "Dietary changes and a doctor visit within a few weeks are recommended."
    ),
    dietary_recommendations=[
        DietaryRecommendation(
            suggestion="Reduce purine-rich foods to lower uric acid",
            reason="Uric acid 8.5 mg/dL exceeds the 3.5-7.2 normal range",
            priority="high",
            foods_to_increase=["water", "cherries", "low-fat dairy"],
            foods_to_avoid=["organ meats", "red meat", "beer"],
        ),
    ],
    lifestyle_modifications=[
        LifestyleModification(
            modification="30 minutes brisk walking 5 days/week",
            reason="Aerobic exercise lowers LDL and improves metabolic health",
            category="exercise",
            priority="medium",
        ),
    ],
    follow_up_actions=[
        FollowUpAction(
            action="Consult GP about elevated uric acid",
            timeframe="Within 2 weeks",
            urgency="soon",
            specialist="General Physician",
        ),
    ],
)


# ── Unit tests — no API calls ─────────────────────────────────

def test_config_valid():
    """GITHUB_API_KEY is present and config validates."""
    is_valid, errors = config.validate_config()
    assert is_valid, f"Config errors: {errors}"


def test_validator_none_input():
    """validate_file() rejects None input."""
    r = validate_file(None)
    assert not r.is_valid


def test_validator_wrong_extension():
    """validate_file() rejects .txt files."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(b"hello")
    tmp.close()
    try:
        r = validate_file(_mock_file(tmp.name))
        assert not r.is_valid
    finally:
        os.unlink(tmp.name)


def test_validator_empty_file():
    """validate_file() rejects empty PDF."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    try:
        r = validate_file(_mock_file(tmp.name))
        assert not r.is_valid
    finally:
        os.unlink(tmp.name)


def test_validator_accepts_valid_pdf():
    """validate_file() accepts a real PDF."""
    if SAMPLE_PDF is None:
        pytest.skip("No sample PDF found at tests/sample_report.pdf")
    r = validate_file(_mock_file(SAMPLE_PDF))
    assert r.is_valid, f"Valid PDF rejected: {r.message}"


def test_validator_size_format_kb():
    """validate_file() reports size in KB for small files."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(b"%PDF-1.4 " + b"x" * 2048)
    tmp.close()
    try:
        r = validate_file(_mock_file(tmp.name))
        size_fmt = r.metadata.get("size_formatted", "")
        assert "KB" in size_fmt, f"Expected KB, got: {size_fmt}"
    finally:
        os.unlink(tmp.name)


def test_validator_wrong_extension_xlsx():
    """validate_file() rejects .xlsx files."""
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.write(b"fake xlsx")
    tmp.close()
    try:
        r = validate_file(_mock_file(tmp.name))
        assert not r.is_valid
    finally:
        os.unlink(tmp.name)


def test_sanitizer_form_feed():
    """sanitize() removes form feed characters."""
    cleaned = sanitize("Hemoglobin\x0c15.3\x0cg/dL")
    assert "\x0c" not in cleaned


def test_sanitizer_hyphenated_line_break():
    """sanitize() joins hyphenated line breaks."""
    cleaned = sanitize("haemo-\nglobin result")
    assert "haemo-\nglobin" not in cleaned


def test_sanitizer_is_meaningful():
    """is_meaningful() correctly classifies text."""
    assert not is_meaningful("")
    assert not is_meaningful("   \n   ")
    assert is_meaningful(
        "Hemoglobin 15.3 g/dL reference range 13.0-17.0 Normal. "
        "WBC 7400 /uL reference range 4000-11000 Normal. "
        "Platelet count 210000 /uL reference range 150000-400000 Normal."
    )


def test_sanitizer_chunk_text():
    """chunk_text() splits long text into multiple chunks."""
    chunks = chunk_text("word " * 5000, max_chars=12000)
    assert len(chunks) > 1
    assert all(len(c) <= 12000 for c in chunks)


def test_sanitizer_get_text_stats():
    """get_text_stats() returns all required fields."""
    stats = get_text_stats("The patient has hemoglobin of 11.2 g/dL.")
    assert "word_count" in stats
    assert "char_count" in stats
    assert "chunk_count" in stats
    assert stats["word_count"] > 0


def test_parser_rejects_wrong_type():
    """parse_document() rejects .txt files."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(b"hello")
    tmp.close()
    try:
        parsed = parse_document(_mock_file(tmp.name))
        assert not parsed.success
    finally:
        os.unlink(tmp.name)


def test_parser_handles_corrupted_pdf():
    """parse_document() handles corrupted PDF gracefully."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(b"this is not a real pdf file content at all")
    tmp.close()
    try:
        parsed = parse_document(_mock_file(tmp.name))
        assert parsed is not None
    finally:
        os.unlink(tmp.name)


async def test_app_handlers():
    """app.py handlers — analyze_report, download_pdf, clear_all."""
    from app import analyze_report, download_pdf, clear_all, _build_history_html

    # clear_all
    initial_state = {
        "analyses_used": 0,
        "last_analysis_time": 0.0,
        "_cache_obj": {},
        "_last_findings": None,
        "_last_recs": None,
        "_history": [],
    }
    try:
        clear_outputs = await clear_all(initial_state)
    except TypeError:
        clear_outputs = await clear_all()
    assert clear_outputs is not None

    # analyze_report with None file
    empty_state = {
        "analyses_used": 0,
        "last_analysis_time": 0.0,
        "_cache_obj": {},
        "_last_findings": None,
        "_last_recs": None,
        "_history": [],
    }
    async for update in analyze_report(None, empty_state):
        status = update[4] if len(update) > 4 else ""
        assert len(status) > 0
        break

    # analyze_report with rate limit reached
    maxed_state = {
        "analyses_used": 2,
        "last_analysis_time": 0.0,
        "_cache_obj": {},
        "_last_findings": None,
        "_last_recs": None,
        "_history": [],
    }
    async for update in analyze_report(None, maxed_state):
        rate_status = update[4] if len(update) > 4 else ""
        assert "session" in rate_status.lower() or "analyses" in rate_status.lower()
        break

    # download_pdf with no findings
    download_pdf({"_last_findings": None, "_last_recs": None})

    # _build_history_html
    state_with_history = {
        "_history": [
            {"filename": "blood_test.pdf", "time": "14:32:01", "urgency": "routine"},
            {"filename": "lab_report.pdf", "time": "14:35:20", "urgency": "consult_soon"},
        ]
    }
    html = _build_history_html(state_with_history)
    assert "blood_test.pdf" in html or "lab_report.pdf" in html
    assert "14:32:01" in html or "14:35:20" in html

    # Newest first
    pos_first = html.find("lab_report.pdf")
    pos_second = html.find("blood_test.pdf")
    assert pos_first < pos_second or pos_first == -1


async def test_rate_limits():
    """Orchestrator enforces session cap and cooldown."""
    orchestrator = MediScanOrchestrator()

    # Session cap
    capped = SessionState(
        analyses_used=MAX_ANALYSES_PER_SESSION,
        last_analysis_time=time.time() - (COOLDOWN_SECONDS + 5),
    )
    async for update in orchestrator.run(None, capped):
        if isinstance(update, AnalysisResult):
            assert not update.success
            assert (
                "session" in update.error.lower()
                or "analyses" in update.error.lower()
            )
            break

    # Cooldown active
    cooling = SessionState(analyses_used=0, last_analysis_time=time.time())
    async for update in orchestrator.run(None, cooling):
        if isinstance(update, AnalysisResult):
            assert not update.success
            assert (
                "wait" in update.error.lower()
                or "seconds" in update.error.lower()
            )
            break


async def test_cache():
    """Cache hit returns result instantly without LLM call."""
    if SAMPLE_PDF is None:
        pytest.skip("No sample PDF found at tests/sample_report.pdf")

    orchestrator = MediScanOrchestrator()
    mock_file = _mock_file(SAMPLE_PDF)
    file_hash = _compute_file_hash(mock_file)

    cached = AnalysisResult(
        success=True,
        findings_md="cached findings",
        recommendations_md="cached recs",
        summary_md="cached summary",
        raw_md="cached raw",
        status="cached status",
        findings=MOCK_FINDINGS,
        recommendations=MOCK_RECS,
    )

    state = SessionState()
    if file_hash:
        state.cache[file_hash] = cached

    t0 = time.time()
    result = None
    async for update in orchestrator.run(mock_file, state):
        if isinstance(update, AnalysisResult):
            result = update
    elapsed = time.time() - t0

    assert result is not None
    assert result.success
    assert result.from_cache
    assert elapsed < 2.0, f"Cache too slow: {elapsed:.2f}s"


async def test_empty_text():
    """analyze_report_text() handles empty text without API call."""
    result = await analyze_report_text(text="", file_name="empty.pdf")
    assert result is not None
    assert result.report_type == "unknown" or not result.lab_values


async def test_non_medical_skip():
    """generate_recommendations() short-circuits for non-medical documents."""
    non_medical = ReportFindings(
        report_type="unknown",
        patient_context=PatientContext(),
        lab_values=[],
        abnormal_flags=[],
        clinical_summary="Insurance document.",
        is_non_medical=True,
        confidence="low",
    )
    t0 = time.time()
    recs = await generate_recommendations(non_medical)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"Non-medical should skip API (took {elapsed:.1f}s)"
    assert len(recs.dietary_recommendations) == 0


# ── LLM tests — real API calls ────────────────────────────────

@pytest.mark.llm
async def test_analyzer():
    """Analyzer agent — real LLM call on sample PDF."""
    if SAMPLE_PDF is None:
        pytest.skip("No sample PDF found at tests/sample_report.pdf")

    parsed = parse_document(_mock_file(SAMPLE_PDF))
    assert parsed.success and parsed.is_meaningful

    result = await analyze_report_text(
        text=parsed.text,
        file_name=parsed.file_name,
        page_count=parsed.page_count,
    )

    assert isinstance(result, ReportFindings)
    assert result.report_type in (
        "lab_report", "clinical_note", "prescription",
        "discharge_summary", "mixed", "unknown",
    )
    assert result.confidence in ("high", "medium", "low")
    assert isinstance(result.lab_values, list)
    assert isinstance(result.abnormal_flags, list)
    assert len(result.clinical_summary) > 20

    findings_md = format_findings_for_display(result)
    assert len(findings_md) > 200


@pytest.mark.llm
async def test_recommender():
    """Recommendation agent — real LLM call on mock findings."""
    result = await generate_recommendations(MOCK_FINDINGS)

    assert isinstance(result, ReportRecommendations)
    assert result.overall_urgency in (
        "routine", "consult_soon", "urgent", "seek_immediate_care"
    )
    assert len(result.overall_assessment) > 30
    assert isinstance(result.dietary_recommendations, list)
    assert isinstance(result.lifestyle_modifications, list)
    assert isinstance(result.follow_up_actions, list)
    assert "AI system" in result.disclaimer

    for diet in result.dietary_recommendations:
        assert hasattr(diet, "suggestion")
        assert hasattr(diet, "reason")
        assert diet.priority in ("high", "medium", "low")

    for fa in result.follow_up_actions:
        assert hasattr(fa, "action")
        assert hasattr(fa, "timeframe")
        assert hasattr(fa, "urgency")

    recs_md = format_recommendations_for_display(result)
    assert len(recs_md) > 200


@pytest.mark.llm
async def test_orchestrator():
    """Full 4-step pipeline — real PDF, real LLM calls."""
    if SAMPLE_PDF is None:
        pytest.skip("No sample PDF found at tests/sample_report.pdf")

    orchestrator = MediScanOrchestrator()
    state = SessionState()
    mock_file = _mock_file(SAMPLE_PDF)

    status_msgs = []
    result = None
    new_state = state

    async for update in orchestrator.run(mock_file, state):
        if isinstance(update, str):
            status_msgs.append(update)
        elif isinstance(update, SessionState):
            new_state = update
        elif isinstance(update, AnalysisResult):
            result = update

    assert result is not None
    assert result.success, f"Pipeline failed: {result.error}"
    assert len(result.findings_md) > 100
    assert len(result.recommendations_md) > 100
    assert len(result.summary_md) > 100
    assert len(result.raw_md) > 50
    assert result.elapsed_seconds > 0
    assert result.findings is not None
    assert result.recommendations is not None
    assert not result.from_cache
    assert new_state.analyses_used == 1
    assert new_state.last_analysis_time > 0
    assert any("Parsing" in m or "parsed" in m.lower() for m in status_msgs)
    assert any("Analyz" in m for m in status_msgs)
    assert any("Recommend" in m for m in status_msgs)


@pytest.mark.llm
async def test_non_medical():
    """Analyzer and recommender handle non-medical documents correctly."""
    insurance_text = """
    HEALTH INSURANCE POLICY DOCUMENT
    Policy Number: HIP/2024/12345
    Sum Insured: Rs. 5,00,000
    Premium Amount: Rs. 12,500 per annum
    Waiting Period: 30 days for general illness.
    Coverage: Hospitalization expenses, ICU charges, ambulance charges.
    Exclusions: Cosmetic surgery, dental treatment, spectacles.
    """

    result = await analyze_report_text(
        text=insurance_text,
        file_name="insurance_policy.pdf",
        page_count=1,
    )
    assert result is not None

    # Force non-medical for recommendation test
    result.is_non_medical = True
    recs = await generate_recommendations(result)
    assert len(recs.dietary_recommendations) == 0
    assert len(recs.lifestyle_modifications) == 0


@pytest.mark.llm
async def test_app_handlers_with_pdf():
    """analyze_report() full flow with a real PDF."""
    if SAMPLE_PDF is None:
        pytest.skip("No sample PDF found at tests/sample_report.pdf")

    from app import analyze_report

    state = {
        "analyses_used": 0,
        "last_analysis_time": 0.0,
        "_cache_obj": {},
        "_last_findings": None,
        "_last_recs": None,
        "_history": [],
    }

    result = None
    async for update in analyze_report(_mock_file(SAMPLE_PDF), state):
        if isinstance(update, tuple) and len(update) >= 5:
            result = update

    assert result is not None
    findings_md, recs_md, summary_md, raw_md, status = result[:5]
    assert len(status) > 0


# ── Standalone script mode ────────────────────────────────────

async def _main():
    print("\n" + "═" * 65)
    print("  MediScan AI — End-to-End QA Test Suite")
    print("═" * 65)

    if SAMPLE_PDF:
        print(f"  PDF  : {SAMPLE_PDF}")
    else:
        print("  PDF  : ⚠️  Not found — LLM tests will be skipped")

    if SAMPLE_DOCX:
        print(f"  DOCX : {SAMPLE_DOCX}")
    else:
        print("  DOCX : ⚠️  Not found — DOCX tests will be skipped")

    print("\n[1] Config...")
    test_config_valid()
    print("   ✅ Config valid")

    print("\n[2] File validator...")
    test_validator_none_input()
    test_validator_wrong_extension()
    test_validator_empty_file()
    test_validator_size_format_kb()
    test_validator_wrong_extension_xlsx()
    if SAMPLE_PDF:
        test_validator_accepts_valid_pdf()
    print("   ✅ Validator correct")

    print("\n[3] Sanitizer...")
    test_sanitizer_form_feed()
    test_sanitizer_hyphenated_line_break()
    test_sanitizer_is_meaningful()
    test_sanitizer_chunk_text()
    test_sanitizer_get_text_stats()
    print("   ✅ Sanitizer correct")

    print("\n[4] Document parser...")
    test_parser_rejects_wrong_type()
    test_parser_handles_corrupted_pdf()
    print("   ✅ Parser edge cases handled")

    print("\n[5] app.py handlers...")
    await test_app_handlers()
    print("   ✅ Handlers correct")

    print("\n[6] Rate limits...")
    await test_rate_limits()
    print("   ✅ Rate limiting correct")

    print("\n[7] Cache...")
    await test_cache()
    print("   ✅ Cache correct")

    print("\n[8] Edge cases...")
    await test_empty_text()
    await test_non_medical_skip()
    print("   ✅ Edge cases handled")

    if SAMPLE_PDF:
        print("\n[9] LLM tests (real API calls)...")
        print("   Running analyzer (15-30s)...")
        await test_analyzer()
        print("   ✅ Analyzer passed")
        print("   Running recommender (10-20s)...")
        await test_recommender()
        print("   ✅ Recommender passed")
        print("   Running full pipeline (25-50s)...")
        await test_orchestrator()
        print("   ✅ Full pipeline passed")

    print("\n" + "═" * 65)
    print("  ✅ ALL TESTS PASSED")
    print("═" * 65)


if __name__ == "__main__":
    asyncio.run(_main())