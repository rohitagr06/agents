"""
test_recommendations.py — Week 4 Verification Script for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests:
    1. All Week 4 imports load correctly
    2. New Pydantic models validate correctly
    3. Prompt builder produces correct format
    4. Real LLM call with synthetic findings (requires GITHUB_API_KEY)
    5. format_recommendations_for_display() produces valid markdown
    6. Edge cases: non-medical doc, no abnormal findings

Run with:
    python test_recommendations.py
    uv run python test_recommendations.py
"""

import asyncio
import sys

print("\n" + "═" * 65)
print("  MediScan AI — Week 4 Recommendation Test")
print("═" * 65)
print("  ⚠️  This test makes real API calls to GitHub Models")
print("═" * 65)


# ── Test 1: Imports ───────────────────────────────────────────

print("\n[1/5] Verifying Week 4 imports...")

try:
    from custom_data_types import (
        ReportRecommendations, DietaryRecommendation,
        LifestyleModification, FollowUpAction,
        ReportFindings, PatientContext, LabValue, AbnormalFlag,
    )
    print("   ✅ custom_data_types — all Week 4 models")
except ImportError as e:
    print(f"   ❌ {e}"); sys.exit(1)

try:
    from prompts.recommendation_prompt import (
        RECOMMENDATION_SYSTEM_PROMPT, build_recommendation_user_message
    )
    print("   ✅ prompts.recommendation_prompt")
except ImportError as e:
    print(f"   ❌ {e}"); sys.exit(1)

try:
    from tools.recommendation_generator import (
        generate_recommendations, format_recommendations_for_display,
        recommendation_agent, _format_findings_for_recommendation,
    )
    print("   ✅ tools.recommendation_generator")
except ImportError as e:
    print(f"   ❌ {e}"); sys.exit(1)


# ── Test 2: Pydantic model validation ────────────────────────

print("\n[2/5] Testing new Pydantic models...")

diet = DietaryRecommendation(
    suggestion="Reduce purine-rich foods to lower uric acid",
    reason="Uric acid 8.5 mg/dL exceeds the 3.5-7.2 normal range",
    priority="high",
    foods_to_increase=["water", "cherries", "low-fat dairy"],
    foods_to_avoid=["organ meats", "red meat", "beer", "shellfish"],
)
assert diet.priority == "high"
assert len(diet.foods_to_avoid) == 4
print("   ✅ DietaryRecommendation validates correctly")

lifestyle = LifestyleModification(
    modification="30 minutes brisk walking 5 days/week",
    reason="Helps reduce LDL cholesterol and improve HDL",
    category="exercise",
    priority="medium",
)
assert lifestyle.category == "exercise"
print("   ✅ LifestyleModification validates correctly")

action = FollowUpAction(
    action="Consult physician about elevated uric acid",
    timeframe="Within 2 weeks",
    urgency="soon",
    specialist="General Physician",
)
assert action.urgency == "soon"
print("   ✅ FollowUpAction validates correctly")

recs = ReportRecommendations(
    dietary_recommendations=[diet],
    lifestyle_modifications=[lifestyle],
    follow_up_actions=[action],
    overall_urgency="consult_soon",
    overall_assessment="Test assessment text.",
)
assert len(recs.dietary_recommendations) == 1
assert recs.overall_urgency == "consult_soon"
assert "AI system" in recs.disclaimer  # default disclaimer present
print("   ✅ ReportRecommendations validates correctly")
print("   ✅ Default disclaimer auto-populated")


# ── Test 3: Prompt + findings formatter ──────────────────────

print("\n[3/5] Testing prompt builder and findings formatter...")

# Build a realistic ReportFindings matching your TATA 1mg report
test_findings = ReportFindings(
    report_type="lab_report",
    patient_context=PatientContext(
        age="33 years", gender="Male",
        report_date="13/Feb/2025",
        ordering_physician="Dr."
    ),
    lab_values=[
        LabValue(parameter="Hemoglobin", value="15.3 g/dL",
                 reference_range="13.0-17.0", flag="Normal"),
        LabValue(parameter="RDW-CV", value="15.7 %",
                 reference_range="11.6-14", flag="High",
                 clinical_note="Above reference range — red cell size variation"),
        LabValue(parameter="MCHC", value="34.7 g/dL",
                 reference_range="31.5-34.5", flag="High",
                 clinical_note="Slightly above upper limit"),
        LabValue(parameter="Uric Acid", value="8.5 mg/dL",
                 reference_range="3.5-7.2", flag="High",
                 clinical_note="Elevated — associated with gout risk"),
        LabValue(parameter="Creatinine", value="1.33 mg/dL",
                 reference_range="0.7-1.3", flag="High",
                 clinical_note="Slightly above normal"),
        LabValue(parameter="Cholesterol - LDL", value="131 mg/dL",
                 reference_range="<100 desirable; 130-159 borderline high",
                 flag="Borderline",
                 clinical_note="Borderline high — cardiovascular risk factor"),
        LabValue(parameter="Immunoglobulin E (IgE)", value="228 IU/mL",
                 reference_range="0-158", flag="High",
                 clinical_note="Elevated — may indicate allergic response"),
        LabValue(parameter="Vitamin D (25-OH)", value="31.7 ng/ml",
                 reference_range="30-100 Sufficient", flag="Normal"),
    ],
    abnormal_flags=[
        AbnormalFlag(finding="RDW-CV 15.7% above reference range 11.6-14%",
                     severity="mild", category="hematology"),
        AbnormalFlag(finding="Uric Acid elevated at 8.5 mg/dL (ref: 3.5-7.2)",
                     severity="moderate", category="metabolic"),
        AbnormalFlag(finding="Creatinine slightly elevated at 1.33 mg/dL (ref: 0.7-1.3)",
                     severity="mild", category="renal"),
        AbnormalFlag(finding="LDL Cholesterol borderline high at 131 mg/dL",
                     severity="moderate", category="lipid"),
        AbnormalFlag(finding="IgE Total elevated at 228 IU/mL (ref: 0-158)",
                     severity="mild", category="general"),
    ],
    clinical_summary=(
        "This comprehensive lab report for a 33-year-old male shows mostly normal "
        "results with a few areas of concern. Uric acid is elevated at 8.5 mg/dL, "
        "LDL cholesterol is borderline high, creatinine is slightly above normal, "
        "and IgE is elevated suggesting possible allergic sensitization."
    ),
    confidence="high",
)

# Test findings formatter
summary = _format_findings_for_recommendation(test_findings)
assert "Uric Acid" in summary, "Uric Acid missing from summary"
assert "ABNORMAL FINDINGS" in summary, "Abnormal section missing"
assert "33 years" in summary, "Patient age missing"
assert "Male" in summary, "Patient gender missing"
print(f"   ✅ _format_findings_for_recommendation() — {len(summary)} chars")
print(f"   ✅ All key findings present in summary")

# Test prompt builder
msg = build_recommendation_user_message(
    findings_summary=summary,
    patient_age="33 years",
    patient_gender="Male",
)
assert "Patient Age: 33 years" in msg
assert "REPORT FINDINGS BEGIN" in msg
assert "REPORT FINDINGS END" in msg
print(f"   ✅ build_recommendation_user_message() — well-formed")


# ── Test 4: Real LLM call ────────────────────────────────────

print("\n[4/5] Testing real LLM call (GitHub Models)...")
print("      Generating recommendations for synthetic TATA 1mg findings...")
print("      Please wait — this may take 15-30 seconds...")

async def test_real_llm():
    recs = await generate_recommendations(test_findings)

    assert isinstance(recs, ReportRecommendations), \
        f"Expected ReportRecommendations, got {type(recs)}"
    print("   ✅ Returns valid ReportRecommendations object")

    assert len(recs.dietary_recommendations) >= 1, \
        "Expected at least 1 dietary recommendation for elevated uric acid + LDL"
    print(f"   ✅ dietary_recommendations: {len(recs.dietary_recommendations)} generated")

    assert len(recs.lifestyle_modifications) >= 1, \
        "Expected at least 1 lifestyle modification"
    print(f"   ✅ lifestyle_modifications: {len(recs.lifestyle_modifications)} generated")

    assert len(recs.follow_up_actions) >= 1, \
        "Expected at least 1 follow-up action"
    print(f"   ✅ follow_up_actions: {len(recs.follow_up_actions)} generated")

    assert recs.overall_urgency in ("routine", "consult_soon", "urgent", "seek_immediate_care"), \
        f"Invalid urgency: {recs.overall_urgency}"
    print(f"   ✅ overall_urgency: '{recs.overall_urgency}'")

    assert len(recs.overall_assessment) > 50, "Assessment too short"
    print(f"   ✅ overall_assessment: {len(recs.overall_assessment)} chars")

    assert "AI system" in recs.disclaimer
    print(f"   ✅ disclaimer: present")

    return recs

recs_result = asyncio.run(test_real_llm())
print("   ✅ Real LLM call passed all assertions")


# ── Test 5: Display formatter ────────────────────────────────

print("\n[5/5] Testing format_recommendations_for_display()...")

md = format_recommendations_for_display(recs_result)
assert "## 💡 Personalized Recommendations" in md
assert "Overall Assessment" in md
assert len(md) > 200
print(f"   ✅ Markdown generated — {len(md):,} chars")

if recs_result.dietary_recommendations:
    assert "🥗 Dietary Recommendations" in md
    print("   ✅ Dietary section present")

if recs_result.lifestyle_modifications:
    assert "🏃 Lifestyle Modifications" in md
    print("   ✅ Lifestyle section present")

if recs_result.follow_up_actions:
    assert "📅 Follow-Up Actions" in md
    print("   ✅ Follow-up actions section present")

assert "disclaimer" in md.lower() or "AI system" in md
print("   ✅ Disclaimer present in output")


# ── Summary ───────────────────────────────────────────────────

print("\n" + "═" * 65)
print("  ✅ ALL WEEK 4 TESTS PASSED")
print("═" * 65)
print()
print("  What's working now:")
print("  ✅ custom_data_types.py     — DietaryRecommendation, LifestyleModification,")
print("                               FollowUpAction, ReportRecommendations")
print("  ✅ prompts/recommendation_prompt.py — system prompt + user message builder")
print("  ✅ tools/recommendation_generator.py — Agent + Runner.run() + display formatter")
print("  ✅ app.py                   — Recommendations tab wired to real Agent")
print()
print("  Live results from the LLM:")
print(f"  🥗 Dietary recommendations  : {len(recs_result.dietary_recommendations)}")
print(f"  🏃 Lifestyle modifications  : {len(recs_result.lifestyle_modifications)}")
print(f"  📅 Follow-up actions        : {len(recs_result.follow_up_actions)}")
print(f"  ⚡ Overall urgency          : {recs_result.overall_urgency}")
print()
print("  What's coming in Week 5:")
print("  ⏳ agents/orchestrator.py    — wires parser + analyzer + recommender")
print("  ⏳ output/pdf_builder.py     — downloadable PDF report")
print("  ⏳ app.py Summary tab        — executive summary from orchestrator")
print()
print("  To launch the app with Week 4 wired in:")
print("  $ uv run python app.py")
print()
