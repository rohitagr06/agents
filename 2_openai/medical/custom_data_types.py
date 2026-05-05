"""
custom_data_types.py — Pydantic Data Models for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
PURPOSE:
    This file defines ALL structured data types used across the
    entire agentic pipeline — exactly like your Deep Research project
    used WebSearchItem, WebSearchPlan, and ReportData.
 
    Every Agent in MediScan AI that has an output_type= parameter
    uses a class defined in THIS file. That means:
 
        report_analyzer_agent   → output_type=ReportFindings
        recommendation_agent    → output_type=ReportRecommendations  (Week 4)
        orchestrator            → uses both above                     (Week 5)
 
WHY A SINGLE CENTRAL FILE FOR ALL DATA TYPES?
    In your Deep Research project, you defined WebSearchItem,
    WebSearchPlan, and ReportData all in one place (custom_data_types.py).
    We follow the exact same pattern here because:
 
    1. SINGLE SOURCE OF TRUTH
       If LabValue has a "flag" field, every agent, every prompt,
       and every UI renderer sees the same definition. There's no
       risk of one file having flag: str and another having flag: bool.
 
    2. PREVENTS CIRCULAR IMPORTS
       Agent files import data types. Data type files should import
       nothing from agent files. Keeping all types here means:
       tools/report_analyzer.py → imports from custom_data_types ✅
       custom_data_types.py     → imports nothing from tools/     ✅
       No circular dependency is possible.
 
    3. EASY TO EVOLVE
       When Week 4 adds ReportRecommendations, we add it here.
       When RC2 adds ImageFinding for radiology reports, we add it here.
       All agents automatically get the updated types.
 
HOW PYDANTIC WORKS WITH THE OPENAI AGENTS SDK:
    When you set output_type=ReportFindings on an Agent, the SDK:
    1. Converts ReportFindings to a JSON Schema and adds it to the
       system prompt as instructions for how to format the response
    2. Parses the LLM's JSON response and validates it against the schema
    3. Returns a proper Python ReportFindings object from Runner.run()
 
    You then access it as:
        result = await Runner.run(report_analyzer_agent, text)
        findings = result.final_output_as(ReportFindings)
        findings.report_type   # "lab_report"
        findings.lab_values    # list[LabValue]
 
    If the LLM returns malformed JSON or missing required fields,
    Pydantic raises a ValidationError automatically — no manual
    checking needed.
 
FIELD DESCRIPTIONS:
    Every field has a description= in its Field() definition.
    This is NOT just documentation — the Agents SDK includes these
    descriptions in the JSON schema sent to the LLM, which directly
    improves the quality of the LLM's output. More specific
    descriptions = more accurate structured output.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────────────────────────
#  LabValue — a single measured parameter from a lab report
#
#  WHY THIS IS A SEPARATE MODEL (not just strings in a list):
#  We could store lab results as List[str] like:
#    ["Hemoglobin: 11.2 g/dL (Low)", "WBC: 7400 /uL (Normal)"]
#  But a structured LabValue object is far more useful because:
#  — The recommendation agent can check value_numeric directly
#    without parsing strings
#  — The UI can colour-code rows by flag status
#  — The PDF builder can format a proper table
#  — Future agents can do math on numeric values
# ─────────────────────────────────────────────────────────────

class LabValue(BaseModel):
    """
    A single measured lab parameter from a medical report.
    Used inside ReportFindings.lab_values list.
    """
    parameter: str = Field(
        description=(
            "The name of the measured parameter. "
            "Examples: 'Hemoglobin', 'WBC Count', 'LDL Cholesterol', "
            "'Fasting Glucose', 'Creatinine', 'Platelet Count'."
        )
    )
    value: str = Field(
        description=(
            "The measured value as a string, including its unit. "
            "Examples: '11.2 g/dL', '7400 /uL', '142 mg/dL', '118 mg/dL'."
        )
    )
    reference_range: str = Field(
        description=(
            "The normal reference range for this parameter, as stated in the report. "
            "Examples: '12.0-17.5 g/dL', '70-99 mg/dL', '<100 mg/dL'. "
            "Use 'Not specified' if the report does not provide a reference range."
        )
    )
    flag: str = Field(
        description=(
            "The status of this value relative to the reference range. "
            "Must be exactly one of: 'Normal', 'Low', 'High', 'Borderline', 'Critical', 'Unknown'. "
            "Use 'Unknown' only if status cannot be determined from the report."
        )
    )
    clinical_note: Optional[str] = Field(
        default=None,
        description=(
            "A brief clinical interpretation of this specific value, if noteworthy. "
            "Example: 'Consistent with mild iron-deficiency anemia.' "
            "Leave null for values that are within normal range."
        )
    )

# ─────────────────────────────────────────────────────────────
#  MedicationItem — a single medication from a prescription
#  or discharge summary
# ─────────────────────────────────────────────────────────────

class MedicationItem(BaseModel):
    """
    A single medication entry from a prescription or discharge summary.
    Used inside ReportFindings.medications list.
    """
    name: str = Field(
        description=(
            "The name of the medication, exactly as written in the report. "
            "Include both brand and generic names if both are present. "
            "Example: 'Metformin 500mg', 'Atorvastatin (Lipitor) 10mg'."
        )
    )
    dosage: str = Field(
        description=(
            "Dosage and frequency as written in the report. "
            "Example: '500mg twice daily', '10mg once at night'. "
            "Use 'Not specified' if not clearly stated."
        )
    )
    purpose: Optional[str] = Field(
        default=None,
        description=(
            "The condition or purpose this medication is prescribed for, if stated. "
            "Example: 'For type 2 diabetes management', 'For cholesterol control'. "
            "Leave null if purpose is not mentioned in the report."
        )
    )

# ─────────────────────────────────────────────────────────────
#  AbnormalFlag — a highlighted abnormal finding
#
#  WHY SEPARATE FROM LabValue?
#  Not all abnormal findings are lab values with numeric results.
#  A clinical note might say "Stage 2 hypertension detected" or
#  "Chest X-ray shows pulmonary infiltrates" — these are important
#  abnormal flags that don't fit the LabValue structure.
#  AbnormalFlag captures ALL concerning findings in one list,
#  whether they are lab-based or clinically observed.
# ─────────────────────────────────────────────────────────────

class AbnormalFlag(BaseModel):
    """
    A single abnormal or concerning finding from any part of the report.
    Used inside ReportFindings.abnormal_flags list.
    """
    finding: str = Field(
        description=(
            "A clear, concise description of the abnormal finding. "
            "Examples: 'Hemoglobin below reference range at 11.2 g/dL', "
            "'Fasting glucose in pre-diabetic range at 118 mg/dL', "
            "'LDL cholesterol significantly elevated at 142 mg/dL'."
        )
    )
    severity: str = Field(
        description=(
            "How serious this finding is. "
            "Must be exactly one of: 'mild', 'moderate', 'severe', 'critical'. "
            "'mild' = slightly outside range, watch and wait. "
            "'moderate' = clearly abnormal, consult physician soon. "
            "'severe' = significantly abnormal, prompt attention needed. "
            "'critical' = immediately dangerous, seek urgent care."
        )
    )
    category: str = Field(
        description=(
            "The medical category this finding belongs to. "
            "Examples: 'hematology', 'metabolic', 'lipid', 'renal', "
            "'hepatic', 'thyroid', 'cardiovascular', 'infectious', 'general'."
        )
    )

# ─────────────────────────────────────────────────────────────
#  PatientContext — basic patient info extracted from the report
#
#  WHY WE EXTRACT THIS:
#  Patient age and gender significantly affect what's considered
#  "normal". A hemoglobin of 11.2 g/dL means different things for
#  a 25-year-old female vs a 60-year-old male. The recommendation
#  agent in Week 4 uses this context to personalize advice.
#
#  WHY ALL FIELDS ARE OPTIONAL:
#  Many medical reports are anonymized or don't include all details.
#  We should never fail because age is missing — we just note it
#  as unknown and the recommendation agent works with what it has.
# ─────────────────────────────────────────────────────────────

class PatientContext(BaseModel):
    """
    Basic patient information extracted from the report header.
    All fields are Optional because reports may omit any of these.
    """
    age: Optional[str] = Field(
        default=None,
        description=(
            "Patient age as found in the report. "
            "Examples: '45 years', '32', 'Not specified'. "
            "Leave null if not mentioned anywhere in the report."
        )
    )
    gender: Optional[str] = Field(
        default=None,
        description=(
            "Patient gender as found in the report. "
            "Examples: 'Male', 'Female', 'Not specified'. "
            "Leave null if not mentioned."
        )
    )
    report_date: Optional[str] = Field(
        default=None,
        description=(
            "The date of the report or sample collection, as written. "
            "Examples: '01/05/2025', 'May 1, 2025', 'Not specified'. "
            "Leave null if no date found."
        )
    )
    ordering_physician: Optional[str] = Field(
        default = None,
        description=(
            "Name of the doctor or physician who ordered the report, if stated. "
            "Leave null if not found."
        )
    )


# ─────────────────────────────────────────────────────────────
#  ReportFindings — the TOP-LEVEL output of the analyzer Agent
#
#  This is the output_type= for report_analyzer_agent.
#  It represents the complete structured extraction of a medical report.
#
#  DESIGN PRINCIPLE — Extraction only, NO advice:
#  This model captures WHAT IS IN THE REPORT — nothing more.
#  The analyzer agent extracts and structures. It does not interpret
#  lifestyle implications, does not suggest diet changes, does not
#  recommend follow-ups. That is the recommendation agent's job (Week 4).
#
#  This clean separation of concerns means:
#  — Extraction accuracy is higher (one focused task)
#  — We can re-run recommendations with different contexts
#    without re-parsing the document
#  — Each agent is independently testable
# ─────────────────────────────────────────────────────────────
# "IMPORTANT: If is_non_medical is true, set this to an empty string ''."

class ReportFindings(BaseModel):
    """
    Complete structured extraction of a medical report.
    This is the output_type of report_analyzer_agent.
 
    Contains everything extracted from the document — lab values,
    medications, abnormal flags, patient context, and a brief summary.
    Does NOT contain recommendations or lifestyle advice — that is
    the job of ReportRecommendations (Week 4).
    """
    report_type: str = Field(
        description=(
            "The type of medical report. "
            "Must be exactly one of: "
            "'lab_report' (blood tests, urine tests, cultures), "
            "'clinical_note' (doctor's observations and diagnoses), "
            "'prescription' (medication orders), "
            "'discharge_summary' (hospital discharge documentation), "
            "'mixed' (report containing multiple types), "
            "'unknown' (cannot determine type from content)."
        )
    )
    patient_context: PatientContext = Field(
        description=(
            "Basic patient information extracted from the report. "
            "Fill in whatever is available — leave fields null if absent."
        )
    )
    lab_values: list[LabValue] = Field(
        default_factory=list,
        description=(
            "List of all measurable lab parameters found in the report. "
            "Include ALL values — both normal and abnormal. "
            "Empty list if the report contains no numeric lab values "
            "(e.g. a clinical note with no bloodwork)."
        )
    )
    medications: list[MedicationItem] = Field(
        default_factory=list,
        description=(
            "List of all medications mentioned in the report. "
            "Include current medications, newly prescribed, and recently stopped. "
            "Empty list if no medications are mentioned."
        )
    )
    abnormal_flags: list[AbnormalFlag] = Field(
        default_factory=list,
        description=(
            "List of ALL concerning or abnormal findings in the report. "
            "Include lab values outside reference range AND clinical observations. "
            "Order by severity: critical first, then severe, moderate, mild. "
            "Empty list if all findings are within normal limits."
        )
    )
    clinical_summary: str = Field(
        description=(
            "A concise 2-4 sentence plain-language summary of the overall report. "
            "Describe what type of report it is, the key findings, and which "
            "areas need attention. Do NOT include lifestyle or dietary advice here — "
            "that belongs in recommendations. "
            "Write for a general audience, not a medical professional."
        )
    )
    is_non_medical: bool = Field(
        default=False,
        description=(
            "Set to true ONLY if the uploaded document is NOT a personal medical report about a patient. "
            "Examples that ARE non-medical (set true): insurance policy documents, health insurance "
            "brochures, policy terms and conditions, legal documents, invoices, news articles, "
            "product information sheets, coverage guides, benefit summaries. "
            "Examples that ARE medical (set false): lab reports, blood test results, "
            "clinical notes, prescriptions, discharge summaries, doctor's letters. "
            "KEY RULE: An insurance policy document describing what conditions are COVERED is NOT "
            "a medical report — it has no patient, no lab values, and no clinical findings. "
            "Set is_non_medical=true for any document that is about insurance, policy, or coverage "
            "rather than about a specific patient's health test results."
        )
    )
    confidence: str = Field(
        default = "high",
        description=(
            "Your confidence in the extraction quality. "
            "Must be one of: 'high', 'medium', 'low'. "
            "'high' = clear, well-structured report with complete data. "
            "'medium' = some ambiguity or missing values, extraction is mostly complete. "
            "'low' = report is poorly structured, heavily redacted, or ambiguous."
        )
    )

## ─────────────────────────────────────────────────────────────
#  DietaryRecommendation — one specific dietary suggestion
#
#  WHY A STRUCTURED MODEL INSTEAD OF A STRING?
#  "Eat more spinach" is vague. A structured model forces the LLM
#  to explain WHY (linked_finding), WHAT specifically (suggestion),
#  and HOW IMPORTANT it is (priority). This makes the UI richer
#  and the advice far more actionable and trustworthy.
# ─────────────────────────────────────────────────────────────

class DietaryRecommendation(BaseModel):
    """One specific dietary suggestion linked to a finding."""
    suggestion: str = Field(
        description=(
            "A specific, actionable dietary suggestion. Be concrete — not "
            "'eat healthy' but 'increase iron-rich foods such as spinach, "
            "lentils, and lean red meat to address low hemoglobin'. "
            "Include specific food examples where possible."
        )
    )
    reason: str = Field(
        description=(
            "Why this dietary change is recommended — which specific lab "
            "finding or condition it addresses. "
            "Example: 'Your RDW-CV of 15.7% and low hemoglobin suggest "
            "possible iron deficiency, which dietary iron can help address.'"
        )
    )
    priority: str = Field(
        description=(
            "How urgently this change should be made. "
            "Must be exactly one of: 'high', 'medium', 'low'. "
            "'high' = start immediately, significant impact expected. "
            "'medium' = implement within a few weeks. "
            "'low' = general wellness improvement, implement when convenient."
        )
    )
    foods_to_increase: list[str] = Field(
        default_factory=list,
        description=(
            "List of specific foods or food groups to increase or add. "
            "Examples: ['spinach', 'lentils', 'fortified cereals', 'lean red meat']. "
            "Empty list if not applicable to this recommendation."
        )
    )
    foods_to_avoid: list[str] = Field(
        default_factory=list,
        description=(
            "List of specific foods or food groups to reduce or avoid. "
            "Examples: ['red meat', 'organ meats', 'beer', 'shellfish']. "
            "Empty list if not applicable to this recommendation."
        )
    )

# ─────────────────────────────────────────────────────────────
#  LifestyleModification — one specific lifestyle suggestion
# ─────────────────────────────────────────────────────────────

class LifestyleModification(BaseModel):
    """One specific lifestyle modification recommendation."""
    modification: str = Field(
        description=(
            "A specific, actionable lifestyle change. Be concrete — not "
            "'exercise more' but '30 minutes of brisk walking 5 days per "
            "week to help reduce LDL cholesterol and improve HDL'. "
            "Include frequency, duration, or measurable targets where possible."
        )
    )
    reason: str = Field(
        description=(
            "Which specific finding this modification addresses and why "
            "this change is expected to help. "
            "Example: 'Your uric acid of 8.5 mg/dL is elevated; reducing "
            "high-purine foods and staying well-hydrated helps the kidneys "
            "excrete uric acid more effectively.'"
        )
    )
    category: str = Field(
        description=(
            "The type of lifestyle change. "
            "Must be exactly one of: 'exercise', 'sleep', 'stress', "
            "'hydration', 'habits', 'monitoring', 'general'."
        )
    )
    priority: str = Field(
        description=(
            "Must be exactly one of: 'high', 'medium', 'low'. "
            "Same scale as DietaryRecommendation.priority."
        )
    )

# ─────────────────────────────────────────────────────────────
#  FollowUpAction — a specific next step the patient should take
# ─────────────────────────────────────────────────────────────

class FollowUpAction(BaseModel):
    """One specific follow-up action the patient should take."""
    action: str = Field(
        description=(
            "A specific next step — a test, appointment, or check the "
            "patient should schedule or perform. "
            "Example: 'Consult your physician about elevated uric acid "
            "(8.5 mg/dL) and discuss whether gout screening is needed.'"
        )
    )
    urgency: str = Field(
        description=(
            "Must be exactly one of: "
            "'routine' (next scheduled visit is fine), "
            "'soon' (within 2-4 weeks), "
            "'urgent' (within 1 week), "
            "'immediate' (seek care today)."
        )
    )
    timeframe: str = Field(
        description=(
            "When this action should be taken. Be specific. "
            "Examples: 'Within 1 week', 'Within 1 month', "
            "'At your next routine check-up (within 3-6 months)', "
            "'Immediately — do not delay'."
        )
    )
    specialist: Optional[str] = Field(
        default=None,
        description=(
            "The type of specialist to consult, if relevant. "
            "Examples: 'Cardiologist', 'Endocrinologist', 'Nephrologist', "
            "'General Physician / Primary Care'. "
            "Leave null if no specific specialist is needed."
        )
    )

# ─────────────────────────────────────────────────────────────
#  ReportRecommendations — the TOP-LEVEL output of the
#  recommendation Agent (Week 4)
#
#  This is output_type= for recommendation_agent.
#  It takes ReportFindings as INPUT and generates personalized
#  advice as OUTPUT.
#
#  DESIGN PRINCIPLE — Advice only, NO extraction:
#  This model contains ZERO extracted lab values. It only contains
#  advice derived from them. The clean separation means:
#  — Recommendation quality is higher (one focused task)
#  — We can regenerate recommendations without re-parsing the PDF
#  — The UI can display findings and recommendations independently
# ─────────────────────────────────────────────────────────────

class ReportRecommendations(BaseModel):
    """
    Personalized recommendations generated from ReportFindings.
    This is the output_type of recommendation_agent.
 
    Contains dietary suggestions, lifestyle modifications, follow-up
    actions, and an overall urgency assessment — all grounded in the
    specific lab findings of the patient. No generic advice.
    """
    overall_urgency: str = Field(
        description=(
            "The overall urgency level considering ALL findings together. "
            "Must be exactly one of: "
            "'routine' (all findings are normal or very mildly off — "
            "no immediate action needed, review at next check-up), "
            "'consult_soon' (some findings need physician attention within "
            "2-4 weeks but no emergency), "
            "'urgent' (one or more findings need attention within 1 week), "
            "'seek_immediate_care' (one or more findings are critically "
            "abnormal — patient should seek care today)."
        )
    )
    overall_assessment: str = Field(
        description=(
            "A plain-language 3-5 sentence overall assessment synthesizing "
            "all findings and recommendations. Write warmly and clearly for "
            "a general audience. Mention the most important concerns first, "
            "acknowledge what is normal, and end with an encouraging but "
            "honest note about the importance of consulting a physician. "
            "Do NOT diagnose. Do NOT prescribe. Do NOT alarm unnecessarily."
        )
    )
    dietary_recommendations: list[DietaryRecommendation] = Field(
        default_factory=list,
        description=(
            "List of specific dietary recommendations grounded in the "
            "patient's actual lab findings. Each recommendation must be "
            "directly linked to a specific abnormal or borderline finding. "
            "Do NOT give generic healthy eating advice — every suggestion "
            "must address a specific value from the report. "
            "Empty list only if ALL findings are perfectly normal."
        )
    )
    lifestyle_modifications: list[LifestyleModification] = Field(
        default_factory=list,
        description=(
            "List of specific lifestyle changes grounded in the patient's "
            "actual findings. Each must be linked to a specific finding. "
            "Cover exercise, sleep, stress management, hydration as relevant. "
            "Do NOT give generic wellness advice — be specific and targeted."
        )
    )
    follow_up_actions: list[FollowUpAction] = Field(
        default_factory=list,
        description=(
            "List of specific next steps the patient should take — "
            "doctor visits, repeat tests, specialist referrals. "
            "Order by urgency: immediate first, routine last. "
            "Every action must be grounded in a specific finding."
        )
    )
    disclaimer: str = Field(
        default=(
            "These recommendations are generated by an AI system for "
            "informational and educational purposes only. They do NOT "
            "constitute professional medical advice, diagnosis, or treatment. "
            "Always consult a qualified healthcare provider before making "
            "any changes to your diet, lifestyle, or medical care."
        ),
        description=(
            "Medical disclaimer. Use the default value — do not modify."
        )
    )