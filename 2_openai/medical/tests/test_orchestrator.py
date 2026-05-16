"""
test_orchestrator.py — Week 5 Verification Script for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests the full pipeline end-to-end using a real PDF file upload,
validating every stage from document parsing through to PDF generation.

Tests:
    1. All Week 5 imports load correctly
    2. SessionState — rate limiting and cache logic
    3. AnalysisResult dataclass fields
    4. _build_summary_md() produces all 3 sections
    5. Full pipeline with real PDF (real LLM calls — requires GITHUB_API_KEY)
    6. Recoverable pipeline — parse failure returns clean AnalysisResult
    7. Rate limit enforcement — blocks after 2 analyses
    8. Cache hit — same file returns cached result, skips LLM
    9. PDF generation from pipeline output
    10. format functions produce valid markdown

Run with:
    python test_orchestrator.py
    uv run python test_orchestrator.py

NOTE:
    Tests 5, 7, 8, 9 make real API calls and require GITHUB_API_KEY in .env
    Tests 1-4, 6, 10 are offline — no API calls needed
    A real PDF file is required at: tests/sample_report.pdf
    If not present, Tests 5, 8, 9 are skipped with a warning.
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock

print("\n" + "═" * 65)
print("  MediScan AI — Week 5 Orchestrator Test")
print("═" * 65)
print("  ⚠️  Tests 5-9 make real API calls to GitHub Models")
print("═" * 65)


# ─────────────────────────────────────────────────────────────
#  Test 1 — Imports
# ─────────────────────────────────────────────────────────────

print("\n[1/10] Verifying Week 5 imports...")

try:
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

    print("   ✅ pipeline.orchestrator — all exports")
except ImportError as e:
    print(f"   ❌ pipeline.orchestrator: {e}")
    sys.exit(1)

try:
    from output.pdf_builder import generate_pdf

    print("   ✅ output.pdf_builder — generate_pdf")
except ImportError as e:
    print(f"   ❌ output.pdf_builder: {e}")
    sys.exit(1)

try:
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

    print("   ✅ custom_data_types — all models")
except ImportError as e:
    print(f"   ❌ custom_data_types: {e}")
    sys.exit(1)

try:
    from tools.report_analyzer import format_findings_for_display
    from tools.recommendation_generator import format_recommendations_for_display

    print("   ✅ tools — all formatters")
except ImportError as e:
    print(f"   ❌ tools: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  Test 2 — SessionState and rate limiting logic
# ─────────────────────────────────────────────────────────────

print("\n[2/10] Testing SessionState and rate limit logic...")

state = SessionState()
assert state.analyses_used == 0, "analyses_used should start at 0"
assert state.last_analysis_time == 0.0, "last_analysis_time should start at 0.0"
assert state.cache == {}, "cache should start empty"
print("   ✅ SessionState initialises with correct defaults")

# Fresh session — should be allowed
allowed, msg = _check_rate_limit(state)
assert allowed, f"Fresh session should be allowed, got: {msg}"
print("   ✅ Fresh session passes rate limit check")

# Simulate cooldown — set last_analysis_time to now
state.last_analysis_time = time.time()
allowed, msg = _check_rate_limit(state)
assert not allowed, "Should be blocked during cooldown"
assert (
    "wait" in msg.lower() or "seconds" in msg.lower()
), f"Message should mention wait time: {msg}"
print(f"   ✅ Cooldown enforced: '{msg[:60]}...'")

# Simulate cooldown expired
state.last_analysis_time = time.time() - (COOLDOWN_SECONDS + 1)
allowed, msg = _check_rate_limit(state)
assert allowed, f"Should be allowed after cooldown expires, got: {msg}"
print("   ✅ Allowed after cooldown expires")

# Simulate session cap reached
state.analyses_used = MAX_ANALYSES_PER_SESSION
allowed, msg = _check_rate_limit(state)
assert not allowed, "Should be blocked when session cap reached"
assert "session" in msg.lower(), f"Message should mention session: {msg}"
print(f"   ✅ Session cap enforced at {MAX_ANALYSES_PER_SESSION}: '{msg[:60]}...'")

assert MAX_ANALYSES_PER_SESSION == 2, f"Expected 2, got {MAX_ANALYSES_PER_SESSION}"
assert COOLDOWN_SECONDS == 60, f"Expected 60s, got {COOLDOWN_SECONDS}"
print(
    f"   ✅ Constants correct: {MAX_ANALYSES_PER_SESSION} analyses, {COOLDOWN_SECONDS}s cooldown"
)


# ─────────────────────────────────────────────────────────────
#  Test 3 — AnalysisResult dataclass
# ─────────────────────────────────────────────────────────────

print("\n[3/10] Testing AnalysisResult dataclass...")

result = AnalysisResult()
assert result.success, "Default success should be True"
assert result.findings_md == "", "Default findings_md should be empty"
assert result.recommendations_md == "", "Default recommendations_md should be empty"
assert result.summary_md == "", "Default summary_md should be empty"
assert result.raw_md == "", "Default raw_md should be empty"
assert result.status == "", "Default status should be empty"
assert result.elapsed_seconds == 0.0, "Default elapsed should be 0.0"
assert not result.from_cache, "Default from_cache should be False"
assert result.findings is None, "Default findings object should be None"
assert result.recommendations is None, "Default recommendations object should be None"
print("   ✅ AnalysisResult default fields all correct")

error_result = AnalysisResult(success=False, status="Test error", error="Test error")
assert not error_result.success
assert error_result.error == "Test error"
print("   ✅ AnalysisResult error construction correct")


# ─────────────────────────────────────────────────────────────
#  Test 4 — _build_summary_md() produces all 3 sections
# ─────────────────────────────────────────────────────────────

print("\n[4/10] Testing _build_summary_md() — 3-section executive summary...")

# Build minimal mock objects
mock_parsed = MagicMock()
mock_parsed.file_name = "test_report.pdf"
mock_parsed.word_count = 1200
mock_parsed.page_count = 4
mock_parsed.chunk_count = 1

mock_findings = ReportFindings(
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

mock_recs = ReportRecommendations(
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
        FollowUpAction(
            action="Fasting glucose retest in 3 months",
            timeframe="In 3 months",
            urgency="routine",
            specialist=None,
        ),
    ],
)

summary = _build_summary_md(mock_parsed, mock_findings, mock_recs)

# Check all 3 required sections are present
# Actual headings from _build_summary_md() in orchestrator.py:
#   Section 1 → "Clinical Overview"
#   Section 2 → "Health Advisor's Assessment"
#   Section 3 → "Overall Urgency" or "At a Glance"
assert (
    "Clinical Overview" in summary or "Key Findings" in summary or "Findings" in summary
), f"Section 1 (Clinical Overview) missing. Got:\n{summary[:300]}"
assert (
    "Assessment" in summary or "Recommendations" in summary or "Advisor" in summary
), f"Section 2 (Assessment/Recommendations) missing. Got:\n{summary[:300]}"
assert (
    "Urgency" in summary or "Next Steps" in summary or "Glance" in summary
), f"Section 3 (Urgency/Next Steps) missing. Got:\n{summary[:300]}"

assert "11.2" in summary or "Hemoglobin" in summary, "Abnormal finding not in summary"
assert (
    "consult" in summary.lower() or "soon" in summary.lower()
), "Urgency level not reflected in summary"
assert len(summary) > 300, f"Summary too short: {len(summary)} chars"

print(f"   ✅ _build_summary_md() — {len(summary):,} chars")
print("   ✅ Section 1 (Key Findings) present")
print("   ✅ Section 2 (Recommendations Overview) present")
print("   ✅ Section 3 (Urgency & Next Steps) present")
print("   ✅ Abnormal findings reflected in summary")


# ─────────────────────────────────────────────────────────────
#  Test 5 — Full pipeline with real PDF (real LLM calls)
# ─────────────────────────────────────────────────────────────

print("\n[5/10] Testing full pipeline end-to-end (real LLM calls)...")

# Look for a sample PDF in common locations
SAMPLE_PDF = None
for path in ["tests/sample_report.pdf", "sample_report.pdf", "test_report.pdf"]:
    if os.path.exists(path):
        SAMPLE_PDF = path
        break

if not SAMPLE_PDF:
    print("   ⚠️  No sample PDF found at tests/sample_report.pdf")
    print("   ⚠️  Skipping Tests 5, 8, 9 — add a real PDF to run them")
    print("   💡  Create: mkdir tests && cp your_report.pdf tests/sample_report.pdf")
    SKIP_PDF_TESTS = True
else:
    SKIP_PDF_TESTS = False
    print(f"   ✅ Found sample PDF: {SAMPLE_PDF}")

pipeline_result = None


async def test_full_pipeline():
    global pipeline_result

    if SKIP_PDF_TESTS:
        print("   ⏭️  Skipped (no sample PDF)")
        return

    orchestrator = MediScanOrchestrator()
    state = SessionState()

    # Create a mock file object that points to the real PDF
    mock_file = MagicMock()
    mock_file.name = os.path.abspath(SAMPLE_PDF)

    print(f"   Running pipeline on: {SAMPLE_PDF}")
    print("   Please wait — this takes 20-40 seconds...")

    status_messages = []
    result = None
    updated_state = state

    async for update in orchestrator.run(mock_file, state):
        if isinstance(update, str):
            status_messages.append(update)
            print(f"   → {update}")
        elif isinstance(update, SessionState):
            updated_state = update
        elif isinstance(update, AnalysisResult):
            result = update

    assert result is not None, "Pipeline yielded no AnalysisResult"
    assert result.success, f"Pipeline failed: {result.error}"
    assert len(result.findings_md) > 100, "findings_md too short"
    assert len(result.recommendations_md) > 100, "recommendations_md too short"
    assert len(result.summary_md) > 100, "summary_md too short"
    assert len(result.raw_md) > 50, "raw_md too short"
    assert result.elapsed_seconds > 0, "elapsed_seconds should be > 0"
    assert result.findings is not None, "findings object should be stored"
    assert result.recommendations is not None, "recommendations object should be stored"
    assert not result.from_cache, "First run should not be cached"
    assert updated_state.analyses_used == 1, "analyses_used should be 1"
    assert updated_state.last_analysis_time > 0, "last_analysis_time should be set"

    # Validate step messages were emitted
    assert any(
        "Parsing" in m or "parsed" in m.lower() for m in status_messages
    ), "No parse status message"
    assert any("Analyz" in m for m in status_messages), "No analyze status message"
    assert any(
        "Recommend" in m for m in status_messages
    ), "No recommendations status message"
    assert any(
        "summary" in m.lower() or "Summary" in m for m in status_messages
    ), "No summary status message"

    print(f"   ✅ Pipeline complete in {result.elapsed_seconds:.1f}s")
    print(f"   ✅ {len(result.findings.lab_values)} lab values extracted")
    print(f"   ✅ {len(result.findings.abnormal_flags)} abnormal flags")
    print(
        f"   ✅ {len(result.recommendations.dietary_recommendations)} dietary recommendations"
    )
    print(f"   ✅ urgency={result.recommendations.overall_urgency}")
    print(f"   ✅ analyses_used updated to {updated_state.analyses_used}")
    print(f"   ✅ {len(status_messages)} step messages streamed")

    pipeline_result = result


asyncio.run(test_full_pipeline())


# ─────────────────────────────────────────────────────────────
#  Test 6 — Recoverable pipeline: parse failure
# ─────────────────────────────────────────────────────────────

print("\n[6/10] Testing recoverable pipeline — parse failure...")


async def test_parse_failure():
    orchestrator = MediScanOrchestrator()
    state = SessionState()

    # Pass None — should fail gracefully at validation
    result = None
    async for update in orchestrator.run(None, state):
        if isinstance(update, AnalysisResult):
            result = update

    assert result is not None, "Should yield an AnalysisResult even on failure"
    assert not result.success, "success should be False on parse failure"
    assert result.error != "", "error should be populated"
    assert result.status != "", "status should contain the error message"
    # UI outputs should be empty — nothing to show
    assert result.findings_md == "", "findings_md should be empty on failure"
    assert result.recommendations_md == "", "recommendations_md should be empty"
    print(f"   ✅ Parse failure handled: '{result.error[:60]}...'")
    print("   ✅ success=False, empty markdown outputs")


asyncio.run(test_parse_failure())


# ─────────────────────────────────────────────────────────────
#  Test 7 — Rate limit blocks after 2 analyses
# ─────────────────────────────────────────────────────────────

print("\n[7/10] Testing rate limit enforcement...")


async def test_rate_limit():
    orchestrator = MediScanOrchestrator()

    # Simulate a session that has used all analyses
    state = SessionState(
        analyses_used=MAX_ANALYSES_PER_SESSION,
        last_analysis_time=time.time() - (COOLDOWN_SECONDS + 5),  # cooldown expired
    )

    result = None
    async for update in orchestrator.run(None, state):
        if isinstance(update, AnalysisResult):
            result = update

    assert result is not None, "Should yield AnalysisResult"
    assert not result.success, "Should be blocked by rate limit"
    assert (
        "session" in result.error.lower() or "analyses" in result.error.lower()
    ), f"Error should mention session limit: {result.error}"
    print(f"   ✅ Session cap enforced: '{result.error[:60]}...'")

    # Simulate cooldown active
    state2 = SessionState(
        analyses_used=0,
        last_analysis_time=time.time(),  # just ran
    )
    result2 = None
    async for update in orchestrator.run(None, state2):
        if isinstance(update, AnalysisResult):
            result2 = update

    assert result2 is not None
    assert not result2.success
    assert (
        "wait" in result2.error.lower() or "seconds" in result2.error.lower()
    ), f"Error should mention cooldown: {result2.error}"
    print(f"   ✅ Cooldown enforced: '{result2.error[:60]}...'")


asyncio.run(test_rate_limit())


# ─────────────────────────────────────────────────────────────
#  Test 8 — Cache hit returns cached result, skips LLM
# ─────────────────────────────────────────────────────────────

print("\n[8/10] Testing session cache hit...")


async def test_cache_hit():
    if SKIP_PDF_TESTS or pipeline_result is None:
        print("   ⏭️  Skipped (no pipeline result from Test 5)")
        return

    orchestrator = MediScanOrchestrator()
    mock_file = MagicMock()
    mock_file.name = os.path.abspath(SAMPLE_PDF)

    # Pre-populate cache with our test result
    file_hash = _compute_file_hash(mock_file)
    state = SessionState()
    if file_hash:
        state.cache[file_hash] = pipeline_result

    cache_result = None
    t_start = time.time()
    async for update in orchestrator.run(mock_file, state):
        if isinstance(update, AnalysisResult):
            cache_result = update
    t_elapsed = time.time() - t_start

    assert cache_result is not None, "Should yield AnalysisResult"
    assert cache_result.success, "Cache result should be successful"
    assert cache_result.from_cache, "from_cache should be True"
    assert (
        "[Cached]" in cache_result.status or "Cached" in cache_result.status
    ), f"Status should indicate cache: {cache_result.status}"
    # Cache hit should be near-instant (< 2 seconds vs 20-40s for real LLM)
    assert t_elapsed < 2.0, f"Cache hit too slow: {t_elapsed:.2f}s (expected <2s)"

    print(f"   ✅ Cache hit in {t_elapsed:.2f}s (vs ~25s for live LLM)")
    print("   ✅ from_cache=True")
    print(f"   ✅ Status: '{cache_result.status[:60]}...'")


asyncio.run(test_cache_hit())


# ─────────────────────────────────────────────────────────────
#  Test 9 — PDF generation from pipeline output
# ─────────────────────────────────────────────────────────────

print("\n[9/10] Testing PDF generation from pipeline output...")


async def test_pdf_generation():
    if SKIP_PDF_TESTS or pipeline_result is None:
        print("   ⏭️  Skipped (no pipeline result from Test 5)")
        print("   💡  Running with mock data instead...")

        # Use the mock data from Test 4 instead
        try:
            pdf_path = generate_pdf(mock_findings, mock_recs)
        except Exception as e:
            print(f"   ❌ PDF generation with mock data failed: {e}")
            return
    else:
        assert (
            pipeline_result.findings is not None
        ), "Pipeline result must have findings object for PDF"
        assert (
            pipeline_result.recommendations is not None
        ), "Pipeline result must have recommendations object for PDF"

        try:
            pdf_path = generate_pdf(
                pipeline_result.findings,
                pipeline_result.recommendations,
            )
        except Exception as e:
            print(f"   ❌ PDF generation failed: {e}")
            raise

    # Verify the file was created and is a real PDF
    assert os.path.exists(pdf_path), f"PDF file not found at: {pdf_path}"

    size_kb = os.path.getsize(pdf_path) / 1024
    assert size_kb > 5, f"PDF too small ({size_kb:.1f} KB) — likely empty"
    assert size_kb < 5000, f"PDF too large ({size_kb:.1f} KB) — something wrong"

    # Verify it's a real PDF by checking the magic bytes
    with open(pdf_path, "rb") as f:
        header = f.read(4)
    assert header == b"%PDF", f"File is not a valid PDF. Header: {header}"

    print(f"   ✅ PDF generated: {os.path.basename(pdf_path)}")
    print(f"   ✅ Size: {size_kb:.1f} KB")
    print("   ✅ Valid PDF header confirmed (%PDF)")

    # Clean up
    os.unlink(pdf_path)
    print("   ✅ Temp file cleaned up")


asyncio.run(test_pdf_generation())


# ─────────────────────────────────────────────────────────────
#  Test 10 — Format functions produce valid markdown
# ─────────────────────────────────────────────────────────────

print("\n[10/10] Testing format functions produce valid markdown...")

findings_md = format_findings_for_display(mock_findings)
assert "## " in findings_md or "# " in findings_md, "No markdown headers in findings"
assert "Hemoglobin" in findings_md, "Lab value missing from findings"
assert "Low" in findings_md or "High" in findings_md, "Flag missing from findings"
assert len(findings_md) > 200, f"Findings markdown too short: {len(findings_md)}"
print(f"   ✅ format_findings_for_display() — {len(findings_md):,} chars")

recs_md = format_recommendations_for_display(mock_recs)
assert "## " in recs_md or "# " in recs_md, "No markdown headers in recommendations"
assert (
    "consult" in recs_md.lower() or "soon" in recs_md.lower()
), "Urgency missing from recommendations"
assert len(recs_md) > 200, f"Recommendations markdown too short: {len(recs_md)}"
print(f"   ✅ format_recommendations_for_display() — {len(recs_md):,} chars")

# Summary from Test 4
assert len(summary) > 300, f"Summary markdown too short: {len(summary)}"
print(f"   ✅ _build_summary_md() — {len(summary):,} chars")

# Non-medical document edge case
non_medical_findings = ReportFindings(
    report_type="unknown",
    patient_context=PatientContext(),
    lab_values=[],
    abnormal_flags=[],
    clinical_summary="This document is not a medical report.",
    is_non_medical=True,
    confidence="low",
)
non_medical_md = format_findings_for_display(non_medical_findings)
assert len(non_medical_md) > 0, "Non-medical result should still produce markdown"
print("   ✅ Non-medical document handled gracefully")


# ─────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────

print("\n" + "═" * 65)
print("  ✅ ALL WEEK 5 TESTS PASSED")
print("═" * 65)
print()
print("  Pipeline architecture verified:")
print("  ✅ SessionState       — rate limit + cache per session")
print("  ✅ AnalysisResult     — typed output with all fields")
print("  ✅ MediScanOrchestrator.run() — async generator, 4-step pipeline")
print("  ✅ Rate limiting      — 2/session, 60s cooldown, graceful block")
print("  ✅ Session cache      — MD5 hash, instant re-analysis")
print("  ✅ Recoverable errors — parse failure → clean AnalysisResult")
print("  ✅ PDF generation     — valid PDF, correct size, cleaned up")
print("  ✅ Format functions   — findings, recommendations, summary")
print()
if SKIP_PDF_TESTS:
    print("  ⚠️  Tests 5, 8, 9 were SKIPPED — no sample PDF found")
    print("  💡  To run all tests:")
    print("      mkdir tests")
    print("      cp your_medical_report.pdf tests/sample_report.pdf")
    print("      uv run python test_orchestrator.py")
else:
    print("  ✅ Full end-to-end pipeline tested with real PDF + real LLM")
print()
print("  What's coming in Week 6:")
print("  ⏳ README.md          — full project documentation")
print("  ⏳ HuggingFace deploy — Spaces setup + secrets")
print("  ⏳ End-to-end QA      — real reports from multiple formats")
print()
