"""
tests/test_analyzer.py — Report Analyzer Agent tests for MediScan AI.
Run standalone: uv run python tests/test_analyzer.py
Run via pytest: uv run python -m pytest tests/test_analyzer.py -v
"""

import asyncio
import pytest

from custom_data_types import ReportFindings, LabValue, AbnormalFlag, PatientContext
from prompts.analyzer_prompt import ANALYZER_SYSTEM_PROMPT, build_analyzer_user_message
from tools.report_analyzer import analyze_report_text, format_findings_for_display


# ── Synthetic lab report used by real LLM tests ───────────────

SYNTHETIC_LAB_REPORT = """
PATIENT LAB REPORT
==================
Patient Name: Test Patient
Age: 45 years
Gender: Male
Report Date: 01/05/2025
Ordering Physician: Dr. Sarah Smith

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

LIPID PANEL
-----------
LDL Cholesterol  142 mg/dL    <100 mg/dL            HIGH
HDL Cholesterol  48 mg/dL     >40 mg/dL             Normal
Total Chol.      210 mg/dL    <200 mg/dL            HIGH
Triglycerides    155 mg/dL    <150 mg/dL            Borderline

PHYSICIAN NOTES
---------------
Patient presents with mild microcytic anemia, likely iron-deficiency.
Fasting glucose in pre-diabetic range. LDL cholesterol elevated.
Recommend dietary modification and 3-month follow-up.
"""


# ── Unit tests — no API calls ─────────────────────────────────

async def test_error_recovery():
    """analyze_report_text() handles empty input gracefully without crashing."""
    result = await analyze_report_text(
        text="",
        file_name="empty.pdf",
        page_count=0,
    )
    assert isinstance(result, ReportFindings), (
        f"Expected ReportFindings on empty input, got {type(result)}"
    )
    assert result.report_type in (
        "unknown", "lab_report", "clinical_note",
        "prescription", "discharge_summary", "mixed",
    ), f"Invalid report_type in recovery: {result.report_type}"


def test_pydantic_lab_value():
    """LabValue creates and validates correctly."""
    lv = LabValue(
        parameter="Hemoglobin",
        value="11.2 g/dL",
        reference_range="12.0-17.5 g/dL",
        flag="Low",
        clinical_note="Below reference range, consistent with mild anemia.",
    )
    assert lv.parameter == "Hemoglobin"
    assert lv.flag == "Low"
    assert lv.clinical_note is not None


def test_pydantic_lab_value_optional_defaults():
    """LabValue optional clinical_note defaults to None."""
    lv = LabValue(
        parameter="WBC Count",
        value="7400 /uL",
        reference_range="4000-11000 /uL",
        flag="Normal",
    )
    assert lv.clinical_note is None


def test_pydantic_abnormal_flag():
    """AbnormalFlag creates and validates correctly."""
    flag = AbnormalFlag(
        finding="LDL cholesterol elevated at 142 mg/dL",
        severity="moderate",
        category="lipid",
    )
    assert flag.severity == "moderate"


def test_pydantic_patient_context_defaults():
    """PatientContext all-null instance is valid."""
    ctx = PatientContext()
    assert ctx.age is None
    assert ctx.gender is None


def test_pydantic_report_findings_defaults():
    """ReportFindings empty list defaults and field defaults are correct."""
    findings = ReportFindings(
        report_type="lab_report",
        patient_context=PatientContext(),
        clinical_summary="Test summary.",
    )
    assert findings.lab_values == []
    assert findings.medications == []
    assert findings.abnormal_flags == []
    assert not findings.is_non_medical
    assert findings.confidence == "high"


def test_system_prompt_content():
    """ANALYZER_SYSTEM_PROMPT has required length and key sections."""
    assert len(ANALYZER_SYSTEM_PROMPT) > 500
    assert "medical document extraction agent" in ANALYZER_SYSTEM_PROMPT.lower()
    assert "DO NOT diagnose" in ANALYZER_SYSTEM_PROMPT
    assert "is_non_medical" in ANALYZER_SYSTEM_PROMPT


def test_user_message_builder():
    """build_analyzer_user_message() includes filename, text, and markers."""
    msg = build_analyzer_user_message(
        extracted_text="Hemoglobin: 11.2 g/dL",
        file_name="blood_test.pdf",
        page_count=2,
    )
    assert "blood_test.pdf" in msg
    assert "Hemoglobin: 11.2 g/dL" in msg
    assert "DOCUMENT TEXT BEGIN" in msg
    assert "DOCUMENT TEXT END" in msg


def test_user_message_builder_chunk_info():
    """build_analyzer_user_message() includes chunk info for multi-chunk docs."""
    msg = build_analyzer_user_message(
        extracted_text="Lab data...",
        file_name="long_report.pdf",
        page_count=10,
        chunk_index=2,
        total_chunks=3,
    )
    assert "chunk 2 of 3" in msg


def test_format_findings_non_medical():
    """format_findings_for_display() shows rejection message for non-medical docs."""
    non_medical = ReportFindings(
        report_type="unknown",
        patient_context=PatientContext(),
        clinical_summary="",
        is_non_medical=True,
    )
    md = format_findings_for_display(non_medical)
    assert "Not a Medical Document" in md


def test_format_findings_structure():
    """format_findings_for_display() produces well-formed markdown."""
    findings = ReportFindings(
        report_type="lab_report",
        patient_context=PatientContext(),
        clinical_summary="Test clinical summary with enough content.",
        lab_values=[
            LabValue(
                parameter="Hemoglobin",
                value="11.2 g/dL",
                reference_range="12.0-17.5 g/dL",
                flag="Low",
            )
        ],
    )
    md = format_findings_for_display(findings)
    assert "## 🔬 Report Analysis" in md
    assert "| Parameter |" in md
    assert len(md) > 200


# ── LLM tests — real API calls ────────────────────────────────

@pytest.mark.llm
async def test_real_llm_call():
    """Full LLM call with synthetic CBC report — validates structured output."""
    findings = await analyze_report_text(
        text=SYNTHETIC_LAB_REPORT,
        file_name="test_cbc_report.pdf",
        page_count=1,
    )

    assert isinstance(findings, ReportFindings), (
        f"Expected ReportFindings, got {type(findings)}"
    )
    assert findings.report_type == "lab_report", (
        f"Expected 'lab_report', got '{findings.report_type}'"
    )
    assert len(findings.lab_values) >= 5, (
        f"Expected at least 5 lab values, got {len(findings.lab_values)}"
    )

    param_names = [lv.parameter.lower() for lv in findings.lab_values]
    assert any("hemoglobin" in p for p in param_names), (
        f"Hemoglobin not found in lab_values. Got: {param_names}"
    )
    assert len(findings.abnormal_flags) >= 1, (
        f"Expected at least 1 abnormal flag, got {len(findings.abnormal_flags)}"
    )
    assert len(findings.clinical_summary) > 50, "clinical_summary too short"
    assert findings.patient_context is not None
    assert not findings.is_non_medical, (
        "is_non_medical should be False for a lab report"
    )
    assert findings.confidence in ("high", "medium", "low"), (
        f"Invalid confidence value: {findings.confidence}"
    )


# ── Standalone script mode ────────────────────────────────────

async def _main():
    print("\n" + "═" * 65)
    print("  MediScan AI — Analyzer Test")
    print("═" * 65)

    print("\n[1/3] Running Pydantic model validation...")
    test_pydantic_lab_value()
    test_pydantic_lab_value_optional_defaults()
    test_pydantic_abnormal_flag()
    test_pydantic_patient_context_defaults()
    test_pydantic_report_findings_defaults()
    print("   ✅ All Pydantic models valid")

    print("\n[2/3] Running prompt builder tests...")
    test_system_prompt_content()
    test_user_message_builder()
    test_user_message_builder_chunk_info()
    print("   ✅ Prompt builder correct")

    print("\n[3/3] Running real LLM call...")
    print("      Please wait — this may take 10-30 seconds...")
    await test_real_llm_call()
    print("   ✅ Real LLM call passed all assertions")

    print("\n" + "═" * 65)
    print("  ✅ ALL TESTS PASSED")
    print("═" * 65)


if __name__ == "__main__":
    asyncio.run(_main())