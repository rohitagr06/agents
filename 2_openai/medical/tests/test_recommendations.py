"""
tests/test_recommendations.py — Recommendation agent tests for MediScan AI.
Run standalone: uv run python tests/test_recommendations.py
Run via pytest: uv run python -m pytest tests/test_recommendations.py -v
"""

import asyncio
import pytest

from custom_data_types import (
    ReportRecommendations,
    DietaryRecommendation,
    LifestyleModification,
    FollowUpAction,
    ReportFindings,
    PatientContext,
    LabValue,
    AbnormalFlag,
)
from prompts.recommendation_prompt import build_recommendation_user_message
from tools.recommendation_generator import (
    generate_recommendations,
    format_recommendations_for_display,
    _format_findings_for_recommendation,
)


# ── Shared test data — no API calls at construction time ──────

TEST_FINDINGS = ReportFindings(
    report_type="lab_report",
    patient_context=PatientContext(
        age="33 years",
        gender="Male",
        report_date="13/Feb/2025",
        ordering_physician="Dr.",
    ),
    lab_values=[
        LabValue(
            parameter="Hemoglobin",
            value="15.3 g/dL",
            reference_range="13.0-17.0",
            flag="Normal",
        ),
        LabValue(
            parameter="RDW-CV",
            value="15.7 %",
            reference_range="11.6-14",
            flag="High",
            clinical_note="Above reference range — red cell size variation",
        ),
        LabValue(
            parameter="Uric Acid",
            value="8.5 mg/dL",
            reference_range="3.5-7.2",
            flag="High",
            clinical_note="Elevated — associated with gout risk",
        ),
        LabValue(
            parameter="Creatinine",
            value="1.33 mg/dL",
            reference_range="0.7-1.3",
            flag="High",
            clinical_note="Slightly above normal",
        ),
        LabValue(
            parameter="Cholesterol - LDL",
            value="131 mg/dL",
            reference_range="<100 desirable; 130-159 borderline high",
            flag="Borderline",
            clinical_note="Borderline high — cardiovascular risk factor",
        ),
        LabValue(
            parameter="Immunoglobulin E (IgE)",
            value="228 IU/mL",
            reference_range="0-158",
            flag="High",
            clinical_note="Elevated — may indicate allergic response",
        ),
    ],
    abnormal_flags=[
        AbnormalFlag(
            finding="Uric Acid elevated at 8.5 mg/dL (ref: 3.5-7.2)",
            severity="moderate",
            category="metabolic",
        ),
        AbnormalFlag(
            finding="Creatinine slightly elevated at 1.33 mg/dL",
            severity="mild",
            category="renal",
        ),
        AbnormalFlag(
            finding="LDL Cholesterol borderline high at 131 mg/dL",
            severity="moderate",
            category="lipid",
        ),
        AbnormalFlag(
            finding="IgE Total elevated at 228 IU/mL (ref: 0-158)",
            severity="mild",
            category="general",
        ),
    ],
    clinical_summary=(
        "This lab report for a 33-year-old male shows mostly normal results "
        "with a few areas of concern. Uric acid is elevated at 8.5 mg/dL, "
        "LDL cholesterol is borderline high, creatinine is slightly above "
        "normal, and IgE is elevated suggesting possible allergic sensitization."
    ),
    confidence="high",
)


# ── Unit tests — no API calls ─────────────────────────────────

def test_dietary_recommendation_validates():
    """DietaryRecommendation creates and validates correctly."""
    diet = DietaryRecommendation(
        suggestion="Reduce purine-rich foods to lower uric acid",
        reason="Uric acid 8.5 mg/dL exceeds the 3.5-7.2 normal range",
        priority="high",
        foods_to_increase=["water", "cherries", "low-fat dairy"],
        foods_to_avoid=["organ meats", "red meat", "beer", "shellfish"],
    )
    assert diet.priority == "high"
    assert len(diet.foods_to_avoid) == 4


def test_lifestyle_modification_validates():
    """LifestyleModification creates and validates correctly."""
    lifestyle = LifestyleModification(
        modification="30 minutes brisk walking 5 days/week",
        reason="Helps reduce LDL cholesterol and improve HDL",
        category="exercise",
        priority="medium",
    )
    assert lifestyle.category == "exercise"


def test_follow_up_action_validates():
    """FollowUpAction creates and validates correctly."""
    action = FollowUpAction(
        action="Consult physician about elevated uric acid",
        timeframe="Within 2 weeks",
        urgency="soon",
        specialist="General Physician",
    )
    assert action.urgency == "soon"


def test_report_recommendations_validates():
    """ReportRecommendations validates and populates default disclaimer."""
    recs = ReportRecommendations(
        dietary_recommendations=[],
        lifestyle_modifications=[],
        follow_up_actions=[],
        overall_urgency="consult_soon",
        overall_assessment="Test assessment.",
    )
    assert recs.overall_urgency == "consult_soon"
    assert "AI system" in recs.disclaimer


def test_format_findings_for_recommendation_content():
    """_format_findings_for_recommendation() includes all key sections."""
    summary = _format_findings_for_recommendation(TEST_FINDINGS)
    assert "Uric Acid" in summary
    assert "ABNORMAL FINDINGS" in summary
    assert "33 years" in summary
    assert "Male" in summary
    assert len(summary) > 100


def test_prompt_builder_structure():
    """build_recommendation_user_message() produces well-formed prompt."""
    summary = _format_findings_for_recommendation(TEST_FINDINGS)
    msg = build_recommendation_user_message(
        findings_summary=summary,
        patient_age="33 years",
        patient_gender="Male",
    )
    assert "Patient Age: 33 years" in msg
    assert "REPORT FINDINGS BEGIN" in msg
    assert "REPORT FINDINGS END" in msg


def test_format_recommendations_for_display_structure():
    """format_recommendations_for_display() produces well-formed markdown."""
    recs = ReportRecommendations(
        dietary_recommendations=[
            DietaryRecommendation(
                suggestion="Reduce purine-rich foods",
                reason="Elevated uric acid",
                priority="high",
                foods_to_increase=["water", "cherries"],
                foods_to_avoid=["organ meats"],
            )
        ],
        lifestyle_modifications=[
            LifestyleModification(
                modification="30 min walking daily",
                reason="Improve cardiovascular health",
                category="exercise",
                priority="medium",
            )
        ],
        follow_up_actions=[
            FollowUpAction(
                action="Repeat uric acid test in 6 weeks",
                timeframe="Within 6 weeks",
                urgency="soon",
            )
        ],
        overall_urgency="consult_soon",
        overall_assessment="Two areas need attention: uric acid and LDL.",
    )
    md = format_recommendations_for_display(recs)
    assert "## 💡 Personalized Recommendations" in md
    assert "Overall Assessment" in md
    assert len(md) > 200
    assert "disclaimer" in md.lower() or "AI system" in md


# ── LLM tests — real API calls ────────────────────────────────

@pytest.mark.llm
async def test_real_llm():
    """Real LLM call — generates recommendations for synthetic findings."""
    await asyncio.sleep(10)  # rate limit recovery after heavy test suite

    recs = await generate_recommendations(TEST_FINDINGS)

    assert isinstance(recs, ReportRecommendations), (
        f"Expected ReportRecommendations, got {type(recs)}"
    )
    assert len(recs.dietary_recommendations) >= 1, (
        "Expected at least 1 dietary recommendation for elevated uric acid + LDL"
    )
    assert len(recs.lifestyle_modifications) >= 1, (
        "Expected at least 1 lifestyle modification"
    )
    assert len(recs.follow_up_actions) >= 1, (
        "Expected at least 1 follow-up action"
    )
    assert recs.overall_urgency in (
        "routine", "consult_soon", "urgent", "seek_immediate_care"
    ), f"Invalid urgency: {recs.overall_urgency}"
    assert len(recs.overall_assessment) > 50, "Assessment too short"
    assert "AI system" in recs.disclaimer


# ── Standalone script mode ────────────────────────────────────

async def _main():
    print("\n" + "═" * 65)
    print("  MediScan AI — Recommendation Agent Test")
    print("═" * 65)

    print("\n[1/4] Pydantic model validation...")
    test_dietary_recommendation_validates()
    test_lifestyle_modification_validates()
    test_follow_up_action_validates()
    test_report_recommendations_validates()
    print("   ✅ All Pydantic models valid")

    print("\n[2/4] Findings formatter and prompt builder...")
    test_format_findings_for_recommendation_content()
    test_prompt_builder_structure()
    print("   ✅ Formatter and prompt builder correct")

    print("\n[3/4] Display formatter...")
    test_format_recommendations_for_display_structure()
    print("   ✅ Markdown output well-formed")

    print("\n[4/4] Real LLM call (requires GITHUB_API_KEY)...")
    print("      Please wait — this may take 15-30 seconds...")
    await test_real_llm()
    print("   ✅ Real LLM call passed all assertions")

    print("\n" + "═" * 65)
    print("  ✅ ALL TESTS PASSED")
    print("═" * 65)


if __name__ == "__main__":
    asyncio.run(_main())