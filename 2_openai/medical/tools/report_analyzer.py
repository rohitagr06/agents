"""
tools/report_analyzer.py — Report Analyzer Agent for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
PURPOSE:
    This is Tool 2 in the agentic pipeline — the first real LLM call
    in MediScan AI. It takes the clean text produced by document_parser.py
    (Week 2) and runs it through an Agent that extracts structured medical
    findings using the OpenAI Agents SDK.
 
    The flow:
        ParsedDocument.text
            ↓
        build_analyzer_user_message()      ← prompts/analyzer_prompt.py
            ↓
        Runner.run(report_analyzer_agent)  ← OpenAI Agents SDK
            ↓
        ReportFindings (Pydantic model)    ← custom_data_types.py
            ↓
        format_findings_for_display()      ← renders to Gradio markdown
 
HOW THIS FITS YOUR DEEP RESEARCH PATTERN:
    Your Deep Research project used:
        planner_agent  → output_type=WebSearchPlan
        search_agent   → tools=[web_search]
        writer_agent   → output_type=ReportData
 
    MediScan RC1 follows the same pattern:
        report_analyzer_agent → output_type=ReportFindings    (this file)
        recommendation_agent  → output_type=ReportRecommendations (Week 4)
        orchestrator          → wires everything together     (Week 5)
 
AGENT vs TOOL vs RUNNER — explained:
 
    Agent:   The configuration object. Defines who the agent IS —
             its name, instructions (system prompt), output_type,
             model, and model_settings. Created ONCE at module level.
             Stateless — safe to reuse across multiple requests.
 
    Runner:  The execution engine. Runner.run(agent, message) actually
             sends the request to the LLM and returns the response.
             Called ONCE per user request inside analyze_report_text().
             Always awaited because it's async.
 
    Tool:    A function the agent can call during its execution.
             The report_analyzer_agent has no tools — it only reads
             and extracts. (The web_search tool in Deep Research is
             an example of a tool used by the search_agent.)
 
ModelSettings — WHY TEMPERATURE=0.1:
    Medical extraction is not creative writing. We want the exact same
    report to produce the exact same output every time we run it.
    Temperature=0.1 (very low) makes the model near-deterministic.
    Contrast with your writer_agent which might use temperature=0.7
    for more natural prose — extraction benefits from low temperature,
    generation benefits from higher temperature.
 
CHUNKING STRATEGY:
    Most medical reports (1-5 pages) fit in one LLM call.
    For longer documents (discharge summaries, full health records),
    the sanitizer splits the text into chunks. We handle this by:
    1. Analyzing the first chunk thoroughly (main findings)
    2. Merging any additional chunk findings into the primary result
    This is simpler than multi-agent chunking for RC1. Week 5 will
    refine this with the orchestrator.
"""
import logging
import asyncio
from agents import Agent, Runner, ModelSettings
from models.models import github_model
from custom_data_types import ReportFindings, LabValue, AbnormalFlag, PatientContext
from prompts.analyzer_prompt import ANALYZER_SYSTEM_PROMPT, build_analyzer_user_message
from utils.sanitizer import chunk_text

logger = logging.getLogger("report_analyzer")

# ── Token budget for a single analyzer LLM call ──────────────
# gpt-4.1-mini hard limit: 8000 tokens total per request.
# Budget: ~1500 system prompt + ~1000 schema + ~1500 output = ~4000 left
# for document text. At 4 chars/token, 12000 chars is the safe ceiling.
# NOTE: sanitizer.py MAX_CHARS_PER_CHUNK defaults to 24000 for general
# use — the analyzer uses a tighter budget because the system prompt +
# schema already consume a large portion of the token window.
SAFE_CHARS_PER_CHUNK: int = 12000


# ─────────────────────────────────────────────────────────────
#  report_analyzer_agent — defined ONCE at module level
#
#  WHY MODULE-LEVEL (not inside a function)?
#  Creating an Agent object is cheap — it's just a config object,
#  not an API call. But defining it at module level means:
#  — It's created once when the module is imported
#  — Every call to analyze_report_text() reuses the same object
#  — No risk of subtle differences between calls
#  — Matches exactly how your Deep Research project defined agents
#    (planner_agent, search_agent, writer_agent all at module level)
#
#  WHY output_type=ReportFindings?
#  This tells the Agents SDK to:
#  1. Include the JSON schema of ReportFindings in the system prompt
#  2. Parse the LLM's response as ReportFindings JSON
#  3. Return a validated Python ReportFindings object
#  If the LLM returns invalid JSON, the SDK retries automatically.
#
#  WHY ModelSettings(temperature=0.1)?
#  Extraction tasks need consistency, not creativity.
#  0.1 makes the model near-deterministic — same input → same output.
# ─────────────────────────────────────────────────────────────

report_analyzer_agent = Agent(
    name="Report Analyzer Agent",
    instructions=ANALYZER_SYSTEM_PROMPT,
    output_type=ReportFindings,
    model=github_model,
    model_settings=ModelSettings(temperature=0.1)
)

# ─────────────────────────────────────────────────────────────
#  _empty_patient_context()  — private helper
#
#  Returns a PatientContext with all null fields.
#  Used in error recovery to avoid None values in the output.
# ─────────────────────────────────────────────────────────────

def _empty_patient_context():
    """Return a PatientContext with all fields set to None."""
    return PatientContext(
        age=None,
        gender=None,
        report_date=None,
        ordering_physician=None
    )

def _merge_findings(all_findings: list[ReportFindings], file_name: str) -> ReportFindings:
    """
    Merge ReportFindings from multiple chunks into one unified result.
 
    MERGING STRATEGY:
    — report_type, patient_context, clinical_summary → from chunk 1
      (it has the header with patient info and the best overall context)
    — lab_values → union of ALL chunks, deduplicated by parameter name
      (case-insensitive). If two chunks extract the same parameter,
      we keep the first occurrence (chunk 1 has priority).
    — medications → same union + dedup strategy by medication name
    — abnormal_flags → union of ALL chunks, deduplicated by finding text
    — is_non_medical → True only if ALL chunks say non-medical
      (one medical chunk means the document is medical)
    — confidence → lowest confidence across all chunks
      (conservative — if any chunk had trouble, report it)
 
    WHY DEDUP BY NAME NOT BY INDEX?
    The same lab parameter (e.g. "Hemoglobin") may appear in multiple
    chunks if it was extracted near a chunk boundary. Deduplication
    by parameter name prevents showing the same value twice in the table.
 
    Args:
        all_findings: List of ReportFindings, one per chunk
        file_name:    Original filename for logging
 
    Returns:
        Single merged ReportFindings with all data combined
    """
    if not all_findings:
        return ReportFindings(
            report_type="unknown",
            patient_context=_empty_patient_context(),
            clinical_summary="No findings could be extracted.",
            confidence="low",
        )
 
    if len(all_findings) == 1:
        return all_findings[0]
 
    # Primary chunk provides the top-level fields
    primary = all_findings[0]
 
    # ── Merge lab_values — deduplicate by parameter name ─────
    seen_params: set[str] = set()
    merged_lab_values = []
    for findings in all_findings:
        for lv in findings.lab_values:
            key = lv.parameter.strip().lower()
            if key not in seen_params:
                seen_params.add(key)
                merged_lab_values.append(lv)
 
    # ── Merge medications — deduplicate by medication name ────
    seen_meds: set[str] = set()
    merged_medications = []
    for findings in all_findings:
        for med in findings.medications:
            key = med.name.strip().lower()
            if key not in seen_meds:
                seen_meds.add(key)
                merged_medications.append(med)
 
    # ── Merge abnormal_flags — deduplicate by finding text ────
    seen_flags: set[str] = set()
    merged_flags = []
    for findings in all_findings:
        for flag in findings.abnormal_flags:
            # Use first 60 chars as dedup key — avoids minor wording differences
            key = flag.finding.strip().lower()[:60]
            if key not in seen_flags:
                seen_flags.add(key)
                merged_flags.append(flag)
 
    # ── Determine overall is_non_medical ─────────────────────
    # True only if ALL chunks say non-medical
    is_non_medical = all(f.is_non_medical for f in all_findings)
 
    # ── Determine overall confidence ─────────────────────────
    # Take the lowest (most conservative) confidence level
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    lowest_confidence = min(
        all_findings,
        key=lambda f: confidence_rank.get(f.confidence, 1)
    ).confidence
 
    # ── Build merged clinical summary ─────────────────────────
    # Use primary chunk's summary — it has the best overall context
    # If the primary is empty, try to find a non-empty one
    clinical_summary = primary.clinical_summary
    if not clinical_summary.strip():
        for f in all_findings[1:]:
            if f.clinical_summary.strip():
                clinical_summary = f.clinical_summary
                break
 
    logger.info(
        f"Merged {len(all_findings)} chunks | "
        f"lab_values={len(merged_lab_values)} | "
        f"medications={len(merged_medications)} | "
        f"flags={len(merged_flags)}"
    )
 
    return ReportFindings(
        report_type=primary.report_type,
        patient_context=primary.patient_context,
        lab_values=merged_lab_values,
        medications=merged_medications,
        abnormal_flags=merged_flags,
        clinical_summary=clinical_summary,
        is_non_medical=is_non_medical,
        confidence=lowest_confidence,
    )

# ─────────────────────────────────────────────────────────────
#  analyze_report_text()  — MAIN PUBLIC ASYNC FUNCTION
#
#  This is what app.py calls. It's async because Runner.run()
#  is a coroutine — it must be awaited.
#
#  It handles the full pipeline:
#  1. Build the user message with document metadata
#  2. Handle multi-chunk documents (if text was split by sanitizer)
#  3. Run the agent via Runner.run()
#  4. Return a validated ReportFindings object
#
#  ERROR HANDLING PHILOSOPHY:
#  This function never raises exceptions to the caller.
#  If anything goes wrong (API error, JSON parse error, timeout),
#  it returns a ReportFindings with is_non_medical=False but
#  clinical_summary containing the error message. This means
#  app.py always gets a ReportFindings back — it never needs
#  to handle exceptions from this function.
# ─────────────────────────────────────────────────────────────

async def analyze_report_text(
    text: str,
    file_name: str = "medical_report",
    page_count: int = 1
    ) -> ReportFindings:
    """
    Run the report analyzer agent on extracted document text.
 
    Takes the clean text from document_parser.py and returns a
    structured ReportFindings object with all lab values, medications,
    abnormal flags, and a clinical summary.
 
    This function is async because Runner.run() is a coroutine.
    Always await this function from the caller.

    MULTI-CHUNK STRATEGY:
    Long reports (like a 15-page TATA 1mg health check) are split into
    multiple chunks by chunk_text(). We analyze ALL chunks in parallel
    using asyncio.gather(), then merge all results into one unified
    ReportFindings. This ensures no lab values from later pages are missed.
 
    Args:
        text:       Cleaned, sanitized text from ParsedDocument.text
        file_name:  Original filename (used in the user message for context)
        page_count: Number of pages (used in the user message for context)
 
    Returns:
        ReportFindings — always returned, never raises.
        On error: returns a ReportFindings with error info in clinical_summary.
 
    Usage in app.py (Week 5):
        findings = await analyze_report_text(
            text=parsed.text,
            file_name=parsed.file_name,
            page_count=parsed.page_count,
        )
    """

    # ── Step 0: Short-circuit on empty input ─────────────────
    # If the parser returned no text (e.g. a scanned image PDF),
    # skip the API call entirely and return a clear error message.
    if not text or not text.strip():
        return ReportFindings(
            report_type="unknown",
            patient_context=_empty_patient_context(),
            lab_values=[],
            medications=[],
            abnormal_flags=[],
            clinical_summary=(
                "No text content was provided for analysis. "
                "The document may be a scanned image PDF with no text layer. "
                "RC2 will support scanned documents via OCR."
            ),
            confidence="low",
        )
 
    # ── Step 1: Handle multi-chunk documents ─────────────────
    # The sanitizer may have split long documents into chunks.
    # We use the first chunk for the primary analysis.
    # chunk_text() with default max_chars returns [full_text] for
    # most medical reports, so this is a no-op in the common case.
    # chunks = chunk_text(text)
    
    chunks = chunk_text(text, max_chars=SAFE_CHARS_PER_CHUNK)
    if not chunks:
        chunks = [text]
    total_chunks = len(chunks)


    logger.info(
        f"Analyzing: {file_name} | "
        f"total_chunks={total_chunks} | "
        f"total_chars={len(text)}"
    )

    # ── Step 2: Analyze ALL chunks ────────────────────────────
    # If only 1 chunk: single API call, no merging needed.
    # If multiple chunks: run all in parallel with asyncio.gather()
    # then merge results. Parallel is faster than sequential for
    # a 15-page report (3 chunks = ~3x speedup).

    async def analyze_single_chunk(chunk_text_: str, chunk_idx: int) -> ReportFindings:
        """Analyze one chunk and return its ReportFindings."""
        user_message = build_analyzer_user_message(
            extracted_text=chunk_text_,
            file_name=file_name,
            page_count=page_count,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
        )
        try:
            logger.info(f"Running chunk {chunk_idx}/{total_chunks} ({len(chunk_text_)} chars)...")
            result = await Runner.run(report_analyzer_agent, user_message)
            findings = result.final_output_as(ReportFindings)
            logger.info(
                f"Chunk {chunk_idx} done | "
                f"lab_values={len(findings.lab_values)} | "
                f"flags={len(findings.abnormal_flags)}"
            )
            return findings
        except Exception as e:
            logger.error(f"Chunk {chunk_idx} failed: {e}")
            # Return empty findings for this chunk — don't crash the whole pipeline
            return ReportFindings(
                report_type="unknown",
                patient_context=_empty_patient_context(),
                clinical_summary="",
                confidence="low",
            )

    try:
        # Run all chunk analyses in parallel
        all_findings: list[ReportFindings] = await asyncio.gather(
            *[analyze_single_chunk(chunk, idx + 1) for idx, chunk in enumerate(chunks)]
        )
 
        # ── Step 3: Merge all chunk results ──────────────────
        # Primary findings come from chunk 1 (has patient context,
        # report type, and clinical summary).
        # All chunks contribute their lab_values, medications,
        # and abnormal_flags — deduplicated by parameter name.
        merged = _merge_findings(all_findings, file_name)
 
        logger.info(
            f"Merge complete | "
            f"report_type={merged.report_type} | "
            f"total_lab_values={len(merged.lab_values)} | "
            f"total_abnormal_flags={len(merged.abnormal_flags)} | "
            f"confidence={merged.confidence}"
        )
        return merged
 
    except Exception as e:
        logger.error(f"Analyzer pipeline failed: {e}")
        return ReportFindings(
            report_type="unknown",
            patient_context=_empty_patient_context(),
            lab_values=[],
            medications=[],
            abnormal_flags=[],
            clinical_summary=(
                f"⚠️ Analysis failed due to an API error: {str(e)[:200]}. "
                f"Please try again. If the problem persists, check your "
                f"GITHUB_API_KEY and internet connection."
            ),
            is_non_medical=False,
            confidence="low",
        )


# ─────────────────────────────────────────────────────────────
#  format_findings_for_display()  — UI renderer
#
#  PURPOSE:
#  Converts a ReportFindings Pydantic object into a rich markdown
#  string for display in the Gradio "Findings" tab.
#
#  WHY HERE AND NOT IN app.py?
#  This function knows the structure of ReportFindings intimately.
#  Keeping it next to the agent that produces ReportFindings means
#  — Single file to edit when ReportFindings schema changes
#  — app.py stays thin (just calls functions, doesn't format data)
#  — Independently testable without launching Gradio
#
#  FLAG EMOJI MAPPING:
#  We map flag strings to coloured emoji indicators so users can
#  scan the findings table at a glance without reading every row.
#  ✅ = Normal (no attention needed)
#  🟡 = Borderline (watch and monitor)
#  🔴 = Low or High (consult physician)
#  🚨 = Critical (seek immediate attention)
# ─────────────────────────────────────────────────────────────

# Maps flag string → display emoji
FLAG_EMOJI: dict[str, str] = {
    "Normal":     "✅ Normal",
    "Low":        "🔴 Low",
    "High":       "🔴 High",
    "Borderline": "🟡 Borderline",
    "Critical":   "🚨 Critical",
    "Unknown":    "❓ Unknown",
}

# Maps severity string → display emoji
SEVERITY_EMOJI: dict[str, str] = {
    "mild":     "🟡 Mild",
    "moderate": "🟠 Moderate",
    "severe":   "🔴 Severe",
    "critical": "🚨 Critical",
}

def format_findings_for_display(findings: ReportFindings) -> str:
    """
    Convert a ReportFindings object into markdown for the Gradio Findings tab.
 
    Args:
        findings: A ReportFindings object from analyze_report_text()
 
    Returns:
        Markdown string ready to display in gr.Markdown()
    """
    # ── Handle non-medical documents ─────────────────────────
    if findings.is_non_medical:
        return (
            "## ❌ Not a Medical Document\n\n"
            "The uploaded file does not appear to be a medical report.\n\n"
            "Please upload a lab report, clinical note, prescription, "
            "or discharge summary in PDF or DOCX format."
        )

    lines: list[str] = []

    # ── Section 1: Report Overview ────────────────────────────
    lines += [
        "## 🔬 Report Analysis",
        "",
        f"**Report Type** `{findings.report_type.replace('_', ' ').title()}` "
        f"&nbsp;&nbsp; **Confidence:** `{findings.confidence.title()}`",
        "",
    ]

    # Patient context (only show fields that are not None)
    ctx = findings.patient_context
    patient_fields = {
        "Age":                ctx.age,
        "Gender":             ctx.gender,
        "Report Date":        ctx.report_date,
        "Ordering Physician": ctx.ordering_physician,
    }
    available = {k: v for k, v in patient_fields.items() if v}
    if available:
        lines.append("### 👤 Patient Information")
        lines.append("")
        for label, value in available.items():
            lines.append(f"- **{label}:** {value}")
        lines.append("")

    # ── Section 2: Clinical Summary ───────────────────────────
    lines += [
        "### 📋 Clinical Summary",
        "",
        findings.clinical_summary,
        "",
    ]
 
    # ── Section 3: Lab Values Table ───────────────────────────
    if findings.lab_values:
        lines += [
            "### 🧪 Laboratory Values",
            "",
            "| Parameter | Result | Reference Range | Status |",
            "|---|---|---|---|",
        ]
        for lv in findings.lab_values:
            emoji = FLAG_EMOJI.get(lv.flag, "❓ Unknown")
            lines.append(
                f"| **{lv.parameter}** | {lv.value} | {lv.reference_range} | {emoji} |"
            )
 
        # Clinical notes for abnormal values (shown below the table)
        notes = [lv for lv in findings.lab_values if lv.clinical_note]
        if notes:
            lines += ["", "**Clinical Notes:**", ""]
            for lv in notes:
                lines.append(f"- **{lv.parameter}:** {lv.clinical_note}")

        lines.append("")
 
    # ── Section 4: Medications ────────────────────────────────
    if findings.medications:
        lines += [
            "### 💊 Medications",
            "",
            "| Medication | Dosage | Purpose |",
            "|---|---|---|",
        ]
        for med in findings.medications:
            purpose = med.purpose or "—"
            lines.append(f"| **{med.name}** | {med.dosage} | {purpose} |")
        lines.append("")
 
    # ── Section 5: Abnormal Flags ─────────────────────────────
    if findings.abnormal_flags:
        lines += [
            "### ⚠️ Areas Requiring Attention",
            "",
        ]
        for flag in findings.abnormal_flags:
            severity_emoji = SEVERITY_EMOJI.get(flag.severity, "🟡 Unknown")
            lines += [
                f"**{severity_emoji}** — {flag.finding}",
                f"*Category: {flag.category.title()}*",
                "",
            ]
    elif not findings.is_non_medical:
        lines += [
            "### ✅ No Abnormal Findings",
            "",
            "All measured values are within normal reference ranges.",
            "",
        ]
 
    # ── Footer ────────────────────────────────────────────────
    lines += [
        "---",
        "*Extracted by MediScan AI · openai/gpt-4.1-mini via AI Models*",
        "*This extraction is for informational purposes only — not medical advice.*",
    ]
 
    return "\n".join(lines)
 