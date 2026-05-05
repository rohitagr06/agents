"""
test_connectivity.py — Week 1 Verification Script
Run this BEFORE anything else to confirm your GitHub Models
token and agent setup are working correctly.

Usage:
    python test_connectivity.py
    # or
    uv run python test_connectivity.py
"""

import asyncio
import sys
from pydantic import BaseModel

print("\n" + "═" * 60)
print("  MediScan AI — Connectivity Test")
print("═" * 60)


# ─────────────────────────────────────────────
#  Step 1 — Config check
# ─────────────────────────────────────────────

print("\n[1/4] Checking environment variables...")
try:
    import config
    is_valid, errors = config.validate_config()
    if not is_valid:
        print("❌ Missing environment variables:")
        for err in errors:
            print(f"   → {err}")
        print("\n💡 Fix: cp .env.example .env  then add your GITHUB_API_KEY")
        sys.exit(1)
    print("   ✅ GITHUB_API_KEY found")
    print(f"   ✅ App: {config.APP_TITLE} {config.APP_VERSION}")
except Exception as e:
    print(f"❌ Could not load config: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Step 2 — Dependency check
# ─────────────────────────────────────────────

print("\n[2/4] Checking installed packages...")
missing = []

packages = [
    ("openai-agents", "agents"),
    ("gradio",        "gradio"),
    ("pymupdf",       "fitz"),
    ("python-docx",   "docx"),
    ("reportlab",     "reportlab"),
    ("pydantic",      "pydantic"),
    ("tenacity",      "tenacity"),
    ("python-dotenv", "dotenv"),
]

for package_name, import_name in packages:
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
    except ImportError:
        print(f"   ❌ {package_name}  ← run: pip install {package_name}")
        missing.append(package_name)

if missing:
    print(f"\n❌ Missing: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Step 3 — GitHub Models connectivity
#  Uses Agent + Runner.run() — your exact pattern
# ─────────────────────────────────────────────

print("\n[3/4] Testing GitHub Models connection...")

from agents import Agent, Runner
from models.models import github_model

class PingResponse(BaseModel):
    message: str

ping_agent = Agent(
    name="Ping Agent",
    instructions="Reply with exactly: { \"message\": \"pong\" } — nothing else.",
    output_type=PingResponse,
    model=github_model,
)

async def test_connection() -> bool:
    try:
        result = await Runner.run(ping_agent, "ping")
        response = result.final_output_as(PingResponse)
        print(f"   ✅ GitHub Models responded")
        print(f"   ✅ Model: openai/gpt-4.1-mini")
        print(f"   ✅ Response: {response.message}")
        return True
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print("   💡 Check your GITHUB_API_KEY in .env")
        return False


# ─────────────────────────────────────────────
#  Step 4 — Medical classification prompt test
#  Validates the model handles medical context
# ─────────────────────────────────────────────

print("\n[4/4] Testing medical classification prompt...")

class ReportClassification(BaseModel):
    report_type: str
    confidence: str

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

async def test_medical_prompt() -> bool:
    try:
        result = await Runner.run(
            classifier_agent,
            (
                "Document contains: Complete Blood Count results showing "
                "hemoglobin 11.2 g/dL, WBC 7,400/μL, platelet count 210,000/μL, "
                "fasting glucose 118 mg/dL, LDL cholesterol 142 mg/dL."
            )
        )
        classification = result.final_output_as(ReportClassification)
        print(f"   ✅ Medical prompt test passed")
        print(f"   ✅ Classified as : {classification.report_type}")
        print(f"   ✅ Confidence    : {classification.confidence}")
        return True
    except Exception as e:
        print(f"   ❌ Medical prompt test failed: {e}")
        return False


# ─────────────────────────────────────────────
#  Run all async tests
# ─────────────────────────────────────────────

async def main():
    conn_ok    = await test_connection()
    if not conn_ok:
        sys.exit(1)
    medical_ok = await test_medical_prompt()
    if not medical_ok:
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  ✅ ALL CHECKS PASSED — Week 1 complete!")
    print("═" * 60)
    print(f"\n  🚀 Ready for Week 2 — Document Parser")
    print(f"  🤖 Model  : openai/gpt-4.1-mini via GitHub Models")
    print(f"  🌐 Base   : https://models.github.ai/inference")
    print(f"  📦 SDK    : openai-agents (Agent + Runner.run)")
    print()

asyncio.run(main())
