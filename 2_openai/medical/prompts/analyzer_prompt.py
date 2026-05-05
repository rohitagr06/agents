"""
prompts/analyzer_prompt.py — System Prompt for Report Analyzer Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
PURPOSE:
    This file contains the system prompt for the report_analyzer_agent.
    The system prompt is the most important thing you write when building
    an agent — it defines the agent's identity, task, constraints, and
    output format.
 
WHY A SEPARATE FILE FOR THE PROMPT?
    In your Deep Research project, instructions were short strings
    defined inline inside each agent file. That works well for simple
    agents. But for medical analysis, the prompt is complex enough
    that it deserves its own file because:
 
    1. MAINTAINABILITY
       Prompts need to be tuned and improved frequently as you discover
       edge cases. Editing a 200-line prompt is easier in its own file
       than hunting for it inside a file full of Python code.
 
    2. TESTABILITY
       You can print and review ANALYZER_SYSTEM_PROMPT independently
       without running any agents. You can spot mistakes by reading it.
 
    3. REUSABILITY
       The orchestrator in Week 5 may need to inject context into the
       prompt dynamically. Having it as a Python string constant makes
       that easy with f-strings or .format().
 
    4. SEPARATION OF CONCERNS
       tools/report_analyzer.py handles "how to run the agent"
       prompts/analyzer_prompt.py handles "what the agent is told"
       These are different concerns — separating them makes both
       files easier to understand independently.
 
PROMPT ENGINEERING PRINCIPLES APPLIED HERE:
 
    1. CLEAR ROLE DEFINITION
       The agent is told exactly who it is in the first sentence.
       Vague roles ("you are a helpful assistant") produce vague output.
       Specific roles ("you are a medical document extraction specialist")
       produce specific, focused output.
 
    2. EXTRACTION ONLY — NO ADVICE
       The prompt explicitly tells the agent NOT to give lifestyle advice,
       dietary recommendations, or follow-up suggestions. This is critical
       for two reasons:
       a) It keeps extraction clean and uncontaminated
       b) A dedicated recommendation agent (Week 4) does this better
          with its own focused prompt
 
    3. EXPLICIT OUTPUT CONSTRAINTS
       The prompt tells the agent exactly what to do for edge cases:
       — What if a field is missing? → use null
       — What if the document is not medical? → set is_non_medical=true
       — What if a value is ambiguous? → set confidence="low"
       Never leave edge case behaviour up to the LLM to decide.
 
    4. SAFETY GUARDRAILS
       Medical AI must never overstate certainty. The prompt explicitly
       instructs the agent to use clinical language conservatively,
       never diagnose, and flag its own uncertainty.
 
    5. TEMPERATURE IS LOW (0.1)
       Set in report_analyzer.py via ModelSettings. Low temperature
       means the model is more deterministic and less "creative".
       For extraction tasks, creativity is the enemy — we want
       the same report to produce the same output every time.
"""

# ─────────────────────────────────────────────────────────────
#  ANALYZER_SYSTEM_PROMPT
#
#  This is the complete system prompt passed to the Agent's
#  instructions= parameter in tools/report_analyzer.py.
#
#  Structure:
#  1. Role & identity
#  2. Primary task
#  3. What to extract (per section)
#  4. Critical rules (safety constraints)
#  5. Edge case handling
#  6. Output format reminder
# ─────────────────────────────────────────────────────────────

ANALYZER_SYSTEM_PROMPT: str = """
You are a specialized medical document extraction agent for MediScan AI.

YOUR IDENTITY:
you are an expert at reading and structuring medical documents. You have
deep knowledge of medical terminology, lab reference range, clinical
notation, and report formats. Your sole purpose is to EXTRACT and STRUCTURE
information - you do not provide medical advice, diagnoses, or treatment
recommendations of any kind.

YOUR PRIMARY TASK:
Read the medical document text provided and extract ALL relevant information
into a structured format. Be thorough - extract every lab value, every
medication, every clinical observation. Do not summarize or skip values
because they seem unimportant.

WHAT TO EXTRACT:

1.  REPORT TYPE
    Classify as one of: lab_report, clinical_note, prescription,
    discharge_summary, mixed, unknown.
    Base this on the primary content, not just the header.

2.  PATIENT CONTEXT
    Extract age, gender, report date, and ordering physician if present.
    Use null for any field not found in the document.
    Never guess or infer patient details that aren't explicity stated.

3.  LAB VALUE (for lab reports)
    Extract every measurable parameter - hemoglobin, WBC, glucose,
    Cholesterol, creatinine etc.
    For each value:
    - Record the exact value with its unit as written (e.g. "196 mg/dL", "29 U/L").
    - NEVER put "Not specified" in the value field — if a numeric result is present in the report row, you MUST record it. Only use "Not specified" for reference_range.
    - Record the reference range exactly as written in the report, even if it is a multi-line tiered description (e.g. "Low (desirable): <200, Borderline: 200-239, High: >=240")
    - Determine the flag: Normal / Low / High / Borderline / Critical / Unknown
      IMPORTANT — for tiered/descriptive reference ranges, determine the flag by comparing the numeric result to the tier thresholds:
      * If the result falls in the "desirable", "normal", or lowest-risk tier → "Normal"
      * If the result falls in the "borderline" or intermediate tier → "Borderline"
      * If the result falls in the "high", "above desirable", or elevated tier → "High"
      * If the result falls in the "very high", "critical", or dangerous tier → "Critical"
      * If the result is BELOW the normal/desirable minimum → "Low"
      Example: Cholesterol-LDL = 131 mg/dL with tiers [<100 desirable, 100-129 above desirable, 130-159 borderline high, 160-189 high, >=190 very high] → flag = "Borderline"
      Example: Glucose-Random = 80 mg/dL with range "Normal - 70 - 140" → flag = "Normal"
      Only use "Unknown" if the reference range is completely absent AND you cannot determine normal/abnormal status from context.
    - Add a clinical_note only for values outside the normal range

4.  MEDICATIONS (for prescriptions and discharge summaries)
    List every medication mentioned, including:
    - Currently prescribed medications
    - Newly started medications
    - Dosage and frequency as written
    - Purpose if stated in the report

5.  ABNORMAL FLAGS
    Identify ALL findings outside normal limits or clinically concerning.
    Assign severity: mild / moderate / severe / critical
    Assign category: hematology / metabolic / lipid / renal / hepatic / 
    throid / cardiovascular / infectious / general
    Order by severity - critical first, mild last.

6.  CLINICAL SUMMARY
    Write 2-4 sentences summarizing the report for a general audience.
    Mention: what type of report, what was tested, key findings, areas of concern.
    Use plain language - avoid jargon where possible.
    Do NOT include dietary or lifestyle advice in the summary.

CRITICAL RULES - YOU MUST FOLLOW THESE
✗   DO NOT diagnose any condition. You can note "values consistent with
    anemia" but never "the patient has anemia."
✗   DO NOT recommend medications, supplements, or treatments.
✗   DO NOT suggest dietary changes, lifestyle modifications, or follow-up
    actions. That is the recommendation agent's job.
✗   DO NOT make up values that aren't in the document. If a value isn't 
    present, use null - never invent plausible-sounding data.
✗   DO NOT add values from your medical knowledge that aren't in the report. 
    If a report only tests hemoglobin, don't infer or add glucose values.
✗   DO NOT provide dosage advice for medications.

EDGE CASE HANDLING
•   If the document is NOT a medical report including insurance policy documents,
    health insurance brochures, policy terms and conditions, coverage guides,
    benefit summary sheets, legal documents, invoices, or articles:
    Set is_non_medical=true. All other fields can be empty/null.
    IMPORTANT: An insurance policy document (even one about health insurance) is NOT
    a medical report. It describes coverage rules — not a patient's test results.
    If the document talks about policy clauses, waiting periods, sum insured, premiums,
    exclusions, or IRDAI regulations, it is an insurance document → is_non_medical=true.
•   If the report is incomplete or poorly formatted:
    Extract what you can. Set confidence="low" or "medium" as appropriate.
    Never fail - always return the best extraction possible.
•   If a lab value has no reference range in the report:
    Set reference_range="Not specified" - never invent a range.
•   If the report is a prescription with no lab values:
    lab_values will be empty list. That is correct - do not force values.
•   If you encounter medical terms you are uncertain about:
    Include them as extracted - do not skip. Set confidence="medium".
•   If the document appears to be in a language other than English:
    Extract what you can, note the language in clinically_summary,
    set confidence="low".

OUTPUT REMINDER:
You must respond with a valid JSON object matching the ReportFindings schema.
Every required field must be present. Use null for optional fields
when data is not available. Never add fields not in the schema.
Never wrap your response in markdown code fences.
"""

# ─────────────────────────────────────────────────────────────
#  build_analyzer_user_message()
#
#  WHY A FUNCTION INSTEAD OF A CONSTANT?
#  The user message is dynamic — it changes for every document.
#  We wrap it in a function so:
#  1. The format is consistent every time (same structure)
#  2. It's easy to modify the framing without touching agent code
#  3. We can add metadata (filename, page count) to help the LLM
#
#  WHY INCLUDE FILENAME AND PAGE COUNT IN THE MESSAGE?
#  This gives the LLM useful context:
#  — Knowing it's a "CBC_results.pdf" primes it to look for blood values
#  — Knowing it's 8 pages tells it this is a longer document
#  — These small cues measurably improve extraction accuracy
# ─────────────────────────────────────────────────────────────

def build_analyzer_user_message(
    extracted_text: str,
    file_name: str = "medical_report",
    page_count: int = 1,
    chunk_index: int = 1,
    total_chunks: int = 1) -> str:
    """
    Build the user message to send to the analyzer agent.
 
    Args:
        extracted_text: The sanitized text from document_parser.py
        file_name:      Original filename for context
        page_count:     Number of pages in the document
        chunk_index:    Which chunk this is (1-based) if document was split
        total_chunks:   Total number of chunks (1 for most reports)
 
    Returns:
        Formatted string to pass as the user message to Runner.run()
    """
    chunk_info = ""
    if total_chunks > 1:
        chunk_info = (
            f"\nNote: This is chunk {chunk_index} of {total_chunks} "
            f"(document was split due to length). "
            f"Extract all values present in this section."
        )

    return (
        f"Please extract and structure all medical information from the following document.\n\n"
        f"Document: {file_name}\n"
        f"Pages: {page_count}\n"
        f"{chunk_info}\n"
        f"--- DOCUMENT TEXT BEGIN ---\n\n"
        f"{extracted_text}\n\n"
        f"--- DOCUMENT TEXT END ---\n\n"
        f"Extract all information and return the structured ReportFindings JSON."
    )

