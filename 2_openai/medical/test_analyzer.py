"""
test_analyzer.py — Week 3 Verification Script for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE:
    Verifies that everything built in Week 3 is working correctly
    before we move to Week 4 (Recommendation Agent). Tests:

        1. All Week 3 imports load without errors
        2. custom_data_types.py — Pydantic models validate correctly
        3. prompts/analyzer_prompt.py — prompt builds correctly
        4. report_analyzer_agent — real LLM call with synthetic lab report
        5. format_findings_for_display() — markdown output is well-formed
        6. Error recovery — graceful handling of edge cases

    ⚠️  REQUIRES YOUR GITHUB_API_KEY IN .env
    Unlike test_parser.py which needed no API key, this test
    makes a real call to GitHub Models (openai/gpt-4.1-mini).
    Make sure your .env has GITHUB_API_KEY set before running.

Run with:
    python test_analyzer.py
    uv run python test_analyzer.py
"""

import asyncio
import sys

print("\n" + "═" * 65)
print("  MediScan AI — Week 3 Analyzer Test")
print("═" * 65)
print("  ⚠️  This test makes real API calls to GitHub Models")
print("  Make sure GITHUB_API_KEY is set in your .env file")
print("═" * 65)


# ─────────────────────────────────────────────────────────────
#  Test 1: Import verification
# ─────────────────────────────────────────────────────────────

print("\n[1/6] Verifying Week 3 imports...")

try:
    from custom_data_types import (
        ReportFindings, LabValue, MedicationItem,
        AbnormalFlag, PatientContext
    )
    print("   ✅ custom_data_types — ReportFindings, LabValue, MedicationItem, AbnormalFlag, PatientContext")
except ImportError as e:
    print(f"   ❌ custom_data_types failed: {e}")
    sys.exit(1)

try:
    from prompts.analyzer_prompt import ANALYZER_SYSTEM_PROMPT, build_analyzer_user_message
    print("   ✅ prompts.analyzer_prompt — ANALYZER_SYSTEM_PROMPT, build_analyzer_user_message")
except ImportError as e:
    print(f"   ❌ prompts.analyzer_prompt failed: {e}")
    sys.exit(1)

try:
    from tools.report_analyzer import (
        analyze_report_text,
        format_findings_for_display,
        report_analyzer_agent,
    )
    print("   ✅ tools.report_analyzer — analyze_report_text, format_findings_for_display, report_analyzer_agent")
except ImportError as e:
    print(f"   ❌ tools.report_analyzer failed: {e}")
    sys.exit(1)

try:
    from agents import Agent, Runner, ModelSettings
    print("   ✅ agents SDK — Agent, Runner, ModelSettings")
except ImportError as e:
    print(f"   ❌ openai-agents SDK not installed: {e}")
    print("      Run: pip install openai-agents")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  Test 2: Pydantic model validation
#  We create model instances directly to verify the schema
#  is correct before making any API calls. This is cheap
#  (no network call) and catches schema bugs early.
# ─────────────────────────────────────────────────────────────

print("\n[2/6] Testing Pydantic model validation...")

# Test 2a: LabValue validates correctly
lv = LabValue(
    parameter="Hemoglobin",
    value="11.2 g/dL",
    reference_range="12.0-17.5 g/dL",
    flag="Low",
    clinical_note="Below reference range, consistent with mild anemia.",
)
assert lv.parameter == "Hemoglobin", "LabValue parameter wrong"
assert lv.flag == "Low",             "LabValue flag wrong"
assert lv.clinical_note is not None, "LabValue clinical_note wrong"
print("   ✅ LabValue — creates and validates correctly")

# Test 2b: LabValue optional field defaults to None
lv_no_note = LabValue(
    parameter="WBC Count",
    value="7400 /uL",
    reference_range="4000-11000 /uL",
    flag="Normal",
)
assert lv_no_note.clinical_note is None, "clinical_note should default to None"
print("   ✅ LabValue — optional clinical_note defaults to None")

# Test 2c: AbnormalFlag validates correctly
flag = AbnormalFlag(
    finding="LDL cholesterol elevated at 142 mg/dL",
    severity="moderate",
    category="lipid",
)
assert flag.severity == "moderate", "AbnormalFlag severity wrong"
print("   ✅ AbnormalFlag — creates and validates correctly")

# Test 2d: PatientContext all-null is valid
ctx = PatientContext()
assert ctx.age is None,   "PatientContext age should default to None"
assert ctx.gender is None, "PatientContext gender should default to None"
print("   ✅ PatientContext — all-null instance is valid")

# Test 2e: ReportFindings empty lists default correctly
findings = ReportFindings(
    report_type="lab_report",
    patient_context=PatientContext(),
    clinical_summary="Test summary.",
)
assert findings.lab_values == [],        "lab_values should default to []"
assert findings.medications == [],       "medications should default to []"
assert findings.abnormal_flags == [],    "abnormal_flags should default to []"
assert findings.is_non_medical == False, "is_non_medical should default to False"
assert findings.confidence == "high",    "confidence should default to 'high'"
print("   ✅ ReportFindings — empty list defaults and field defaults correct")

print("   ✅ All Pydantic model validations passed")


# ─────────────────────────────────────────────────────────────
#  Test 3: Prompt builder
# ─────────────────────────────────────────────────────────────

print("\n[3/6] Testing prompt builder...")

# Test 3a: ANALYZER_SYSTEM_PROMPT is non-empty and contains key sections
assert len(ANALYZER_SYSTEM_PROMPT) > 500, "System prompt seems too short"
assert "medical document extraction agent" in ANALYZER_SYSTEM_PROMPT.lower(), \
    "System prompt missing role definition"
assert "DO NOT diagnose" in ANALYZER_SYSTEM_PROMPT, \
    "System prompt missing safety guardrails"
assert "is_non_medical" in ANALYZER_SYSTEM_PROMPT, \
    "System prompt missing edge case handling"
print(f"   ✅ ANALYZER_SYSTEM_PROMPT — {len(ANALYZER_SYSTEM_PROMPT)} chars, all key sections present")

# Test 3b: build_analyzer_user_message() produces correct format
msg = build_analyzer_user_message(
    extracted_text="Hemoglobin: 11.2 g/dL",
    file_name="blood_test.pdf",
    page_count=2,
)
assert "blood_test.pdf" in msg,           "Filename missing from user message"
assert "Hemoglobin: 11.2 g/dL" in msg,   "Text missing from user message"
assert "DOCUMENT TEXT BEGIN" in msg,      "Section markers missing"
assert "DOCUMENT TEXT END" in msg,        "Section markers missing"
print("   ✅ build_analyzer_user_message() — filename, text, markers all present")

# Test 3c: Multi-chunk message includes chunk info
msg_chunked = build_analyzer_user_message(
    extracted_text="Lab data...",
    file_name="long_report.pdf",
    page_count=10,
    chunk_index=2,
    total_chunks=3,
)
assert "chunk 2 of 3" in msg_chunked, "Chunk info missing from multi-chunk message"
print("   ✅ build_analyzer_user_message() — chunk info included for multi-chunk docs")


# ─────────────────────────────────────────────────────────────
#  Test 4: Real LLM call — the heart of Week 3
#
#  We send a realistic synthetic CBC lab report to the agent
#  and verify the structured output is correct.
#  This makes a real API call to GitHub Models.
# ─────────────────────────────────────────────────────────────

print("\n[4/6] Testing real LLM call (GitHub Models API)...")
print("      Sending synthetic CBC lab report to report_analyzer_agent...")
print("      Please wait — this may take 10-30 seconds...")

SYNTHETIC_LAB_REPORT = """
PATIENT LAB REPORT
==================
Patient Name: Test Patient
Age: 45 years
Gender: Male
Report Date: 01/05/2025
Ordering Physician: Dr. Sarah Smith
Lab ID: LAB-2025-TEST-001

COMPLETE BLOOD COUNT (CBC)
--------------------------
Test             Result       Reference Range      Flag
Hemoglobin       11.2 g/dL    12.0-17.5 g/dL       LOW
WBC Count        7400 /uL     4000-11000 /uL        Normal
Platelet Count   210000 /uL   150000-400000 /uL     Normal
Hematocrit       34%          36-46%                LOW
MCV              78 fL        80-100 fL             LOW

METABOLIC PANEL
---------------
Fasting Glucose  118 mg/dL    70-99 mg/dL           HIGH
Creatinine       0.9 mg/dL    0.6-1.2 mg/dL         Normal
Sodium           139 mEq/L    136-145 mEq/L          Normal

LIPID PANEL
-----------
LDL Cholesterol  142 mg/dL    <100 mg/dL            HIGH
HDL Cholesterol  48 mg/dL     >40 mg/dL              Normal
Total Chol.      210 mg/dL    <200 mg/dL             HIGH
Triglycerides    155 mg/dL    <150 mg/dL             Borderline

PHYSICIAN NOTES
---------------
Patient presents with mild microcytic anemia, likely iron-deficiency.
Fasting glucose in pre-diabetic range. LDL cholesterol elevated.
Recommend dietary modification and 3-month follow-up.
"""

async def test_real_llm_call():
    """Run the actual LLM call and validate the structured output."""

    findings = await analyze_report_text(
        text=SYNTHETIC_LAB_REPORT,
        file_name="test_cbc_report.pdf",
        page_count=1,
    )

    # ── Verify the ReportFindings structure ──────────────────
    assert isinstance(findings, ReportFindings), \
        f"Expected ReportFindings, got {type(findings)}"
    print("   ✅ Returns a valid ReportFindings object")

    # Report type
    assert findings.report_type == "lab_report", \
        f"Expected 'lab_report', got '{findings.report_type}'"
    print(f"   ✅ report_type = '{findings.report_type}'")

    # Lab values extracted
    assert len(findings.lab_values) >= 5, \
        f"Expected at least 5 lab values, got {len(findings.lab_values)}"
    print(f"   ✅ lab_values = {len(findings.lab_values)} values extracted")

    # Check key parameters are present
    param_names = [lv.parameter.lower() for lv in findings.lab_values]
    assert any("hemoglobin" in p for p in param_names), \
        f"Hemoglobin not found in lab_values. Got: {param_names}"
    print(f"   ✅ 'Hemoglobin' found in extracted lab values")

    # Check abnormal flags detected
    assert len(findings.abnormal_flags) >= 1, \
        f"Expected at least 1 abnormal flag, got {len(findings.abnormal_flags)}"
    print(f"   ✅ abnormal_flags = {len(findings.abnormal_flags)} flags detected")

    # Clinical summary is non-empty
    assert len(findings.clinical_summary) > 50, \
        "clinical_summary too short"
    print(f"   ✅ clinical_summary = {len(findings.clinical_summary)} chars")

    # Patient context
    assert findings.patient_context is not None, "patient_context is None"
    print(f"   ✅ patient_context extracted (age={findings.patient_context.age})")

    # Not marked as non-medical
    assert not findings.is_non_medical, \
        "is_non_medical should be False for a lab report"
    print(f"   ✅ is_non_medical = False (correctly identified as medical)")

    # Confidence
    assert findings.confidence in ("high", "medium", "low"), \
        f"Invalid confidence value: {findings.confidence}"
    print(f"   ✅ confidence = '{findings.confidence}'")

    return findings

findings_result = asyncio.run(test_real_llm_call())
print("   ✅ Real LLM call passed all assertions")


# ─────────────────────────────────────────────────────────────
#  Test 5: format_findings_for_display()
# ─────────────────────────────────────────────────────────────

print("\n[5/6] Testing format_findings_for_display()...")

md = format_findings_for_display(findings_result)

assert "## 🔬 Report Analysis" in md,   "Missing report header"
assert "lab_values" not in md,           "Raw field names should not appear in output"
assert "| Parameter |" in md or len(findings_result.lab_values) == 0, \
    "Lab values table missing"
assert len(md) > 200, "Display markdown seems too short"

print(f"   ✅ Markdown output generated — {len(md)} chars")
print(f"   ✅ Report header present")
print(f"   ✅ Lab values table rendered")

# Test non-medical document display
non_medical = ReportFindings(
    report_type="unknown",
    patient_context=PatientContext(),
    clinical_summary="",
    is_non_medical=True,
)
non_medical_md = format_findings_for_display(non_medical)
assert "Not a Medical Document" in non_medical_md, \
    "Non-medical rejection message missing"
print("   ✅ Non-medical document correctly shows rejection message")


# ─────────────────────────────────────────────────────────────
#  Test 6: Error recovery
#  We can't easily force a real API error, but we verify the
#  error recovery path produces a valid ReportFindings object
#  by testing the fallback directly.
# ─────────────────────────────────────────────────────────────

print("\n[6/6] Testing error recovery path...")

async def test_error_recovery():
    """Test that analyze_report_text() handles empty input gracefully."""
    # Empty text — the agent should still return something valid
    result = await analyze_report_text(
        text="",
        file_name="empty.pdf",
        page_count=0,
    )
    # Should return a ReportFindings (not crash)
    assert isinstance(result, ReportFindings), \
        "Error recovery should return ReportFindings, not raise"
    assert result.report_type in ("unknown", "lab_report", "clinical_note",
                                   "prescription", "discharge_summary", "mixed"), \
        f"Invalid report_type in recovery: {result.report_type}"
    print(f"   ✅ Empty input handled gracefully — type='{result.report_type}'")

asyncio.run(test_error_recovery())


# ─────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────

print("\n" + "═" * 65)
print("  ✅ ALL WEEK 3 TESTS PASSED")
print("═" * 65)
print()
print("  What's working now:")
print("  ✅ custom_data_types.py       — ReportFindings + nested Pydantic models")
print("  ✅ prompts/analyzer_prompt.py — system prompt + user message builder")
print("  ✅ tools/report_analyzer.py   — Agent + Runner.run() + display formatter")
print("  ✅ app.py                     — Findings tab wired to real Agent output")
print()
print("  Live results from the LLM:")
print(f"  📋 Report type    : {findings_result.report_type}")
print(f"  🧪 Lab values     : {len(findings_result.lab_values)} extracted")
print(f"  ⚠️  Abnormal flags : {len(findings_result.abnormal_flags)} detected")
print(f"  🎯 Confidence     : {findings_result.confidence}")
print()
print("  What's coming in Week 4:")
print("  ⏳ prompts/recommendation_prompt.py  — system prompt for advisor Agent")
print("  ⏳ custom_data_types.py update       — ReportRecommendations model")
print("  ⏳ tools/recommendation_generator.py — Recommendation Agent + Runner.run()")
print()
print("  To launch the app with Week 3 wired in:")
print("  $ python app.py")
print("  $ uv run python app.py")
print()
