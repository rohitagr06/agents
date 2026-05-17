"""
tests/test_connectivity.py — GitHub Models connectivity tests.
Run standalone: uv run python tests/test_connectivity.py
Run via pytest: uv run python -m pytest tests/test_connectivity.py -v
"""

import asyncio
import sys
import pytest
from pydantic import BaseModel


# ── Agents and model imported at module level is fine —
#    no API calls happen at import time, only at test time
from agents import Agent, Runner
from models.models import github_model


# ── Agents defined at module level — same as your tool files.
#    Creating an Agent object is just a config object, no API call.

class PingResponse(BaseModel):
    message: str


class ReportClassification(BaseModel):
    report_type: str
    confidence: str


ping_agent = Agent(
    name="Ping Agent",
    instructions='Reply with exactly: { "message": "pong" } — nothing else.',
    output_type=PingResponse,
    model=github_model,
)

classifier_agent = Agent(
    name="Classifier Agent",
    instructions=(
        "You are a medical document classifier. "
        "Given a document description, classify it as one of: "
        "lab_report, clinical_note, prescription, discharge_summary, unknown. "
        "Also rate your confidence as: high, medium, or low."
    ),
    output_type=ReportClassification,
    model=github_model,
)


# ── Tests ──────────────────────────────────────────────────────

@pytest.mark.llm
async def test_connection():
    """Verify GitHub Models API key and basic connectivity."""
    result = await Runner.run(ping_agent, "ping")
    response = result.final_output_as(PingResponse)
    assert response.message, "Expected a non-empty message from ping agent"


@pytest.mark.llm
async def test_medical_prompt():
    """Verify the model correctly classifies a lab report description."""
    result = await Runner.run(
        classifier_agent,
        (
            "Document contains: Complete Blood Count results showing "
            "hemoglobin 11.2 g/dL, WBC 7,400/μL, platelet count 210,000/μL, "
            "fasting glucose 118 mg/dL, LDL cholesterol 142 mg/dL."
        ),
    )
    classification = result.final_output_as(ReportClassification)
    assert classification.report_type in (
        "lab_report", "clinical_note", "prescription",
        "discharge_summary", "unknown"
    ), f"Unexpected report_type: {classification.report_type}"
    assert classification.confidence in (
        "high", "medium", "low"
    ), f"Unexpected confidence: {classification.confidence}"


# ── Standalone script mode ─────────────────────────────────────

async def _main():
    """Run connectivity checks as a standalone script with friendly output."""
    print("\n" + "═" * 60)
    print("  MediScan AI — Connectivity Test")
    print("═" * 60)

    print("\n[1/4] Checking environment variables...")
    import config
    is_valid, errors = config.validate_config()
    if not is_valid:
        for err in errors:
            print(f"   ❌ {err}")
        print("\n💡 Fix: cp .env.example .env  then add your GITHUB_API_KEY")
        sys.exit(1)
    print(f"   ✅ GITHUB_API_KEY found")
    print(f"   ✅ App: {config.APP_TITLE} {config.APP_VERSION}")

    print("\n[2/4] Checking installed packages...")
    packages = [
        ("openai-agents", "agents"),
        ("gradio", "gradio"),
        ("pymupdf", "fitz"),
        ("python-docx", "docx"),
        ("reportlab", "reportlab"),
        ("pydantic", "pydantic"),
        ("python-dotenv", "dotenv"),
    ]
    missing = []
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name}")
            missing.append(package_name)
    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        sys.exit(1)

    print("\n[3/4] Testing GitHub Models connection...")
    try:
        result = await Runner.run(ping_agent, "ping")
        response = result.final_output_as(PingResponse)
        print(f"   ✅ GitHub Models responded: {response.message}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        sys.exit(1)

    print("\n[4/4] Testing medical classification prompt...")
    try:
        result = await Runner.run(
            classifier_agent,
            "Complete Blood Count: hemoglobin 11.2 g/dL, WBC 7400/μL"
        )
        classification = result.final_output_as(ReportClassification)
        print(f"   ✅ Classified as: {classification.report_type}")
        print(f"   ✅ Confidence   : {classification.confidence}")
    except Exception as e:
        print(f"   ❌ Medical prompt failed: {e}")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  ✅ ALL CHECKS PASSED")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(_main())