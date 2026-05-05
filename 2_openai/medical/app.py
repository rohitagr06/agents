"""
app.py — MediScan AI — Main Gradio Application
RC1: One-shot medical report analysis with stunning UI.

Stack:
  - Gradio 4.x for UI
  - openai-agents SDK (Agent + Runner.run) for LLM pipeline
  - GitHub Models (openai/gpt-4.1-mini) via AsyncOpenAI

Run with:
    python app.py
    uv run python app.py
"""

import asyncio
import gradio as gr
import config
from models.models import github_model
from tools.document_parser import parse_document, format_parsed_for_display
from tools.report_analyzer import analyze_report_text, format_findings_for_display
from tools.recommendation_generator import generate_recommendations, format_recommendations_for_display

# ─────────────────────────────────────────────
#  Custom CSS — Deep Medical Aesthetic
#  Clean navy/teal palette, refined typography,
#  smooth micro-interactions, premium card feel
# ─────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --navy:        #0B1929;
    --navy-mid:    #112236;
    --navy-light:  #1A3350;
    --teal:        #00C4B4;
    --teal-dim:    #009E90;
    --teal-glow:   rgba(0, 196, 180, 0.15);
    --sky:         #4FC3F7;
    --amber:       #FFB347;
    --danger:      #FF6B6B;
    --success:     #69E0A5;
    --text-bright: #F0F6FF;
    --text-mid:    #A8C0D6;
    --text-dim:    #5A7A96;
    --card-bg:     rgba(17, 34, 54, 0.85);
    --card-border: rgba(0, 196, 180, 0.18);
    --radius-lg:   16px;
    --radius-md:   10px;
    --shadow-card: 0 8px 40px rgba(0,0,0,0.45), 0 1px 0 rgba(0,196,180,0.1);
    --shadow-glow: 0 0 30px rgba(0, 196, 180, 0.12);
}

/* ── Base & Background ── */
body, .gradio-container {
    background: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-bright) !important;
    min-height: 100vh;
}

.gradio-container {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,196,180,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 90% 80%, rgba(79,195,247,0.05) 0%, transparent 50%),
        var(--navy) !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 16px 60px !important;
}

/* ── Header ── */
.medi-header {
    text-align: center;
    padding: 52px 20px 36px;
    position: relative;
}

.medi-header::after {
    content: '';
    display: block;
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    margin: 20px auto 0;
}

.medi-logo-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 10px;
}

.medi-icon {
    font-size: 2.6rem;
    filter: drop-shadow(0 0 12px rgba(0,196,180,0.6));
}

.medi-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 0%, var(--teal) 60%, var(--sky) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1 !important;
    margin: 0 !important;
}

.medi-subtitle {
    font-size: 1.05rem;
    color: var(--text-mid);
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-top: 6px;
}

.medi-version-badge {
    display: inline-block;
    background: rgba(0,196,180,0.1);
    border: 1px solid rgba(0,196,180,0.3);
    color: var(--teal);
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Disclaimer Banner ── */
.disclaimer-banner {
    background: linear-gradient(135deg, rgba(255,107,107,0.08), rgba(255,179,71,0.06));
    border: 1px solid rgba(255,107,107,0.25);
    border-radius: var(--radius-md);
    padding: 12px 18px;
    margin: 0 0 28px 0;
    font-size: 0.82rem;
    color: #FFB9B9;
    line-height: 1.55;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}

.disclaimer-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

/* ── Section Labels ── */
.section-label {
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--card-border), transparent);
}

/* ── Cards ── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    border-color: rgba(0,196,180,0.3);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}

/* ── Upload Zone ── */
.upload-zone .wrap {
    background: rgba(11,25,41,0.6) !important;
    border: 2px dashed rgba(0,196,180,0.3) !important;
    border-radius: var(--radius-lg) !important;
    transition: all 0.3s ease !important;
    min-height: 160px !important;
}

.upload-zone .wrap:hover {
    border-color: var(--teal) !important;
    background: rgba(0,196,180,0.04) !important;
}

.upload-zone .icon-wrap svg { color: var(--teal) !important; }

.upload-zone .wrap span {
    color: var(--text-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Analyze Button ── */
.analyze-btn button {
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dim) 100%) !important;
    color: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    padding: 14px 32px !important;
    border-radius: var(--radius-md) !important;
    border: none !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(0,196,180,0.25) !important;
    position: relative;
    overflow: hidden;
}

.analyze-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,196,180,0.4) !important;
}

.analyze-btn button:active {
    transform: translateY(0) !important;
}

.analyze-btn button:disabled {
    opacity: 0.5 !important;
    transform: none !important;
    cursor: not-allowed !important;
}

/* ── Clear Button ── */
.clear-btn button {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    border: 1px solid rgba(90,122,150,0.3) !important;
    border-radius: var(--radius-md) !important;
    width: 100% !important;
    padding: 12px !important;
    transition: all 0.2s ease !important;
}

.clear-btn button:hover {
    border-color: var(--text-dim) !important;
    color: var(--text-mid) !important;
    background: rgba(90,122,150,0.08) !important;
}

/* ── Status Box ── */
.status-box {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    min-height: 110px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.status-box textarea, .status-box input {
    background: transparent !important;
    color: var(--text-mid) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
    padding: 0 20px !important;
    border: none !important;
    resize: none !important;
    box-shadow: none !important;
    height: auto !important;
    overflow: hidden !important;
    width: 100% !important;
    outline: none !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    display: block !important;
    align-self: center !important;
}

/* ── Output Tabs ── */
.output-tabs .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid var(--card-border) !important;
    gap: 4px !important;
    padding-bottom: 0 !important;
}

.output-tabs .tab-nav button {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.output-tabs .tab-nav button.selected {
    background: rgba(0,196,180,0.1) !important;
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
}

.output-tabs .tab-nav button:hover:not(.selected) {
    color: var(--text-mid) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── Output Markdown ── */
.output-content, .output-content * {
    color: var(--text-bright) !important;
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.7 !important;
}

.output-content h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.3rem !important;
    color: var(--teal) !important;
    border-bottom: 1px solid var(--card-border) !important;
    padding-bottom: 6px !important;
    margin-top: 24px !important;
    margin-bottom: 12px !important;
}

.output-content h3 {
    font-size: 0.95rem !important;
    color: var(--sky) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-weight: 600 !important;
}

.output-content strong { color: #FFE082 !important; }
.output-content code {
    background: rgba(0,196,180,0.1) !important;
    color: var(--teal) !important;
    font-family: 'DM Mono', monospace !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
    font-size: 0.88em !important;
}

.output-content ul li::marker { color: var(--teal) !important; }
.output-content ol li::marker { color: var(--teal) !important; }

/* ── Download Button ── */
.download-btn button {
    background: transparent !important;
    color: var(--sky) !important;
    border: 1px solid rgba(79,195,247,0.3) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}

.download-btn button:hover {
    background: rgba(79,195,247,0.08) !important;
    border-color: var(--sky) !important;
}

/* ── How It Works ── */
.steps-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 28px 0;
}

.step-card {
    background: rgba(11,25,41,0.7);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 16px 12px;
    text-align: center;
    position: relative;
}

.step-num {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, var(--teal), var(--teal-dim));
    color: var(--navy);
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px;
}

.step-icon { font-size: 1.4rem; margin-bottom: 6px; }
.step-title { font-size: 0.8rem; font-weight: 600; color: var(--text-bright); margin-bottom: 4px; }
.step-desc { font-size: 0.72rem; color: var(--text-dim); line-height: 1.4; }

/* ── Footer ── */
.medi-footer {
    text-align: center;
    padding: 28px 0 8px;
    border-top: 1px solid var(--card-border);
    color: var(--text-dim);
    font-size: 0.78rem;
}

.medi-footer a { color: var(--teal); text-decoration: none; }

/* ── Gradio overrides ── */
.gr-form, .gr-box, .gr-panel {
    background: transparent !important;
    border: none !important;
}

label.svelte-1b6s6s { color: var(--text-mid) !important; font-family: 'DM Sans' !important; }

.generating {
    background: linear-gradient(90deg, var(--teal-glow) 25%, transparent 37%, var(--teal-glow) 63%) !important;
    background-size: 400% 100% !important;
    animation: shimmer 1.4s ease infinite !important;
}

@keyframes shimmer {
    0% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--navy-mid); }
::-webkit-scrollbar-thumb { background: var(--navy-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal-dim); }
"""


# ─────────────────────────────────────────────
#  HTML Blocks
# ─────────────────────────────────────────────

HEADER_HTML = f"""
<div class="medi-header">
  <div class="medi-logo-row">
    <span class="medi-icon">🫀</span>
    <h1 class="medi-title">MediScan AI</h1>
  </div>
  <p class="medi-subtitle">Intelligent Medical Report Analyzer — Powered by DipsAI</p>
  <span class="medi-version-badge">{config.APP_VERSION}</span>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer-banner">
  <span class="disclaimer-icon">⚠️</span>
  <span>
    <strong>Medical Disclaimer:</strong> This analysis is AI-generated for informational purposes only.
    It does <strong>not</strong> constitute professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider. In emergencies, call your local emergency number immediately.
  </span>
</div>
"""

HOW_IT_WORKS_HTML = """
<div class="steps-row">
  <div class="step-card">
    <div class="step-num">1</div>
    <div class="step-icon">📄</div>
    <div class="step-title">Upload Report</div>
    <div class="step-desc">Upload your PDF or DOCX medical document</div>
  </div>
  <div class="step-card">
    <div class="step-num">2</div>
    <div class="step-icon">🔍</div>
    <div class="step-title">Extract & Parse</div>
    <div class="step-desc">AI reads and structures the document content</div>
  </div>
  <div class="step-card">
    <div class="step-num">3</div>
    <div class="step-icon">🧠</div>
    <div class="step-title">Analyze</div>
    <div class="step-desc">Agent identifies findings, flags, and patterns</div>
  </div>
  <div class="step-card">
    <div class="step-num">4</div>
    <div class="step-icon">💡</div>
    <div class="step-title">Get Insights</div>
    <div class="step-desc">Receive diet, lifestyle & follow-up guidance</div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="medi-footer">
  MediScan AI &nbsp;·&nbsp; DipsAI &nbsp;·&nbsp; Built with AI Models + OpenAI Agents SDK + Gradio
  &nbsp;·&nbsp; For educational use only
</div>
"""


# ─────────────────────────────────────────────
#  Core Analysis Function
#
#  RC1 SHELL — async def, matching the pattern
#  that Runner.run() will plug into in Week 3.
#
#  Week 2: document_parser tool replaces raw_md
#  Week 3: report_analyzer Agent fills findings
#  Week 4: recommendation Agent fills recommendations
#  Week 5: orchestrator wires all agents together
# ─────────────────────────────────────────────

async def analyze_report(file) -> tuple[str, str, str, str, str]:
    """
    Main analysis pipeline — called by Gradio on button click.
    async because Runner.run() (openai-agents) is fully async.

    Week 2 state:
        ✅ File validation        — validator.py
        ✅ PDF/DOCX extraction    — document_parser.py (PyMuPDF + python-docx)
        ✅ Text sanitization      — sanitizer.py
        ✅ Raw text tab populated — real extracted content
        ✅ Findings tab           — placeholder until Week 3 Agent
        ✅ Recommendations tab    — placeholder until Week 4 Agent
        ⏳ Summary tab            — placeholder until Week 5 orchestrator

    Returns:
        findings_md        : str — structured findings markdown
        recommendations_md : str — diet/lifestyle/follow-up markdown
        summary_md         : str — executive summary markdown
        raw_md             : str — extracted raw text markdown
        status_msg         : str — status bar message
    """
    # ── Step 1: Parse the document ───────────────────────────
    # parse_document() internally calls validate_file() first,
    # then routes to _parse_pdf() or _parse_docx() based on extension.
    # It always returns a ParsedDocument — never raises exceptions.
    # We don't need our own None/extension guards anymore —
    # the parser handles all of that and returns clear error messages.
    parsed = parse_document(file)

    # ── Step 2: Handle parsing failure ──────────────────────
    # If validation failed (no file, wrong type, too big, corrupted)
    # OR if the PDF/DOCX library threw an error,
    # parsed.success will be False and parsed.error has the message.
    if not parsed.success:
        return ("", "", "", "", parsed.error)

    # ── Step 3: Check text quality ───────────────────────────
    # Even if parsing succeeded, the text might be too short to analyze
    # (e.g. a scanned image PDF with no text layer).
    # We show the raw tab with what we got, and surface the warning.
    if not parsed.is_meaningful:
        warning_message = parsed.warning or (
            "⚠️ Extracted text is too short to analyze meaningfully. "
            "The document may be a scanned image. RC2 will support OCR."
        )
        raw_md = format_parsed_for_display(parsed)
        return ("", "", "", raw_md, warning_message)

    # ── Step 4: Build the Raw Text tab output ────────────────
    # format_parsed_for_display() formats the ParsedDocument into
    # a clean markdown string showing file metadata + extracted content.
    # This is the REAL extracted text — no more placeholders here.

    raw_md = format_parsed_for_display(parsed)

    # ── Step 5: Run the Report Analyzer Agent ────────────────
    # This is the first real LLM call in MediScan AI.
    # analyze_report_text() calls Runner.run(report_analyzer_agent, ...)
    # and returns a validated ReportFindings Pydantic object.
    # format_findings_for_display() converts it to markdown for the UI.


    findings = await analyze_report_text(
        text=parsed.text,
        file_name=parsed.file_name,
        page_count=parsed.page_count
    )
    findings_md = format_findings_for_display(findings)

    # ── Step 6: Run the Recommendation Agent ─────────────────
    # generate_recommendations() takes ReportFindings from Step 5
    # and runs Runner.run(recommendation_agent, ...) to produce
    # personalized dietary, lifestyle, and follow-up advice.
    recommendations = await generate_recommendations(findings)
    recommendations_md = format_recommendations_for_display(recommendations)

    # ── Step 7: Summary placeholder (Week 5) ─────────────────
    summary_md = f"""
## 📋 Executive Summary
 
> ⏳ **Week 5** — Orchestrator Agent synthesizes findings + recommendations into a summary here.
 
### Document Stats
- **{parsed.word_count:,} words** extracted from **{parsed.page_count} page(s)**
- Text split into **{parsed.chunk_count} LLM chunk(s)** for processing
- File: `{parsed.file_name}`
 
---
*Analysis generated by MediScan AI {config.APP_VERSION} · GitHub Models (openai/gpt-4.1-mini) · For informational use only*
*{config.MEDICAL_DISCLAIMER}*
"""

    # ── Step 8: Status message ────────────────────────────────
    size_info = f"{parsed.word_count:,} words . {parsed.page_count} page(s)"
    status = f"✅ Parsed successfully — {parsed.file_name} ({size_info})"

    # Surface any non-fatal warnings (e.g. partial scanned pages)
    if parsed.warning:
        status = f"{status} · {parsed.warning}"
    return findings_md, recommendations_md, summary_md, raw_md, status


async def clear_all():
    """Reset all outputs. async to match analyze_report pattern."""
    return None, "", "", "", "", "Ready. Upload a report to begin."


# ─────────────────────────────────────────────
#  Build Gradio UI
# ─────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="MediScan AI — Medical Report Analyzer",
        theme=gr.themes.Base(
            primary_hue="teal",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
        ),
    ) as app:

        # ── Header ──
        gr.HTML(HEADER_HTML)
        gr.HTML(DISCLAIMER_HTML)
        gr.HTML(HOW_IT_WORKS_HTML)

        # ── Main Layout ──
        with gr.Row(equal_height=False):

            # ── LEFT COLUMN — Upload & Controls ──
            with gr.Column(scale=4, min_width=300):

                gr.HTML('<div class="section-label">Upload Document</div>')

                file_input = gr.File(
                    label="",
                    file_types=[".pdf", ".docx", ".doc"],
                    file_count="single",
                    elem_classes=["upload-zone"],
                    height=170,
                )

                gr.HTML("""
                <div style="font-size:0.75rem; color:#5A7A96; text-align:center; margin: -8px 0 14px; line-height:1.5;">
                  Supported formats: PDF · DOCX · DOC &nbsp;|&nbsp; Max size: 10 MB<br>
                  Your document is processed in-session and never stored.
                </div>
                """)

                analyze_btn = gr.Button(
                    "🔬  Analyze Report",
                    variant="primary",
                    elem_classes=["analyze-btn"],
                )

                gr.HTML('<div style="height:8px;"></div>')

                clear_btn = gr.Button(
                    "✕  Clear",
                    elem_classes=["clear-btn"],
                )

                gr.HTML('<div style="height:20px;"></div>')
                gr.HTML('<div class="section-label">Status</div>')

                status_box = gr.Textbox(
                    value="Ready. Upload a report to begin.",
                    show_label=False,
                    label=None,
                    interactive=False,
                    lines=3,
                    elem_classes=["status-box"],
                    container=False,
                )

                # ── Info Box ──
                gr.HTML("""
                <div style="margin-top: 20px; background: rgba(0,196,180,0.04);
                     border: 1px solid rgba(0,196,180,0.12); border-radius: 10px; padding: 16px;">
                  <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                       color:#00C4B4; font-family:'DM Mono',monospace; margin-bottom:10px;">
                    Supported Report Types
                  </div>
                  <div style="font-size:0.82rem; color:#A8C0D6; line-height:2;">
                    🧪 Lab Reports (CBC, Metabolic, Lipid)<br>
                    🩺 Clinical Diagnosis Notes<br>
                    💊 Prescriptions<br>
                    🏥 Discharge Summaries
                  </div>
                </div>
                """)

                # ── Model Info Box ──
                gr.HTML("""
                <div style="margin-top: 12px; background: rgba(79,195,247,0.04);
                     border: 1px solid rgba(79,195,247,0.12); border-radius: 10px; padding: 14px;">
                  <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                       color:#4FC3F7; font-family:'DM Mono',monospace; margin-bottom:8px;">
                    Powered By
                  </div>
                  <div style="font-size:0.78rem; color:#A8C0D6; line-height:1.9;">
                    🤖 openai/gpt-4.1-mini<br>
                    ⚡ AI Models API<br>
                    🔗 OpenAI Agents SDK
                  </div>
                </div>
                """)

            # ── RIGHT COLUMN — Output ──
            with gr.Column(scale=8, min_width=500):

                gr.HTML('<div class="section-label">Analysis Results</div>')

                with gr.Tabs(elem_classes=["output-tabs"]):

                    with gr.TabItem("🔬 Findings"):
                        findings_output = gr.Markdown(
                            value="*Upload a report and click Analyze to see findings here.*",
                            elem_classes=["output-content"],
                        )

                    with gr.TabItem("💡 Recommendations"):
                        recommendations_output = gr.Markdown(
                            value="*Personalized diet, lifestyle, and follow-up suggestions will appear here.*",
                            elem_classes=["output-content"],
                        )

                    with gr.TabItem("📋 Summary"):
                        summary_output = gr.Markdown(
                            value="*An executive summary of key findings will appear here.*",
                            elem_classes=["output-content"],
                        )

                    with gr.TabItem("📄 Raw Extracted Text"):
                        raw_output = gr.Markdown(
                            value="*The raw text extracted from your document will appear here.*",
                            elem_classes=["output-content"],
                        )

        # ── Footer ──
        gr.HTML(FOOTER_HTML)

        # ── Event Handlers ──
        # analyze_report is async — Gradio 4.x handles async fn natively
        analyze_btn.click(
            fn=analyze_report,
            inputs=[file_input],
            outputs=[
                findings_output,
                recommendations_output,
                summary_output,
                raw_output,
                status_box,
            ],
            show_progress="full",
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                file_input,
                findings_output,
                recommendations_output,
                summary_output,
                raw_output,
                status_box,
            ],
        )

    return app


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Hard stop if GITHUB_API_KEY is missing
    is_valid, errors = config.validate_config()
    if not is_valid:
        print("\n❌ Cannot start — missing required config:")
        for err in errors:
            print(f"   → {err}")
        print("\n💡 Fix: cp .env.example .env  then add your GITHUB_API_KEY\n")
        raise SystemExit(1)

    print(f"\n🫀 Starting {config.APP_TITLE} {config.APP_VERSION}")
    print(f"   Model       : openai/gpt-4.1-mini via AI Models")
    print(f"   SDK         : openai-agents (Agent + Runner.run)")
    print(f"   Environment : {config.APP_ENV}")
    print(f"   Port        : {config.GRADIO_SERVER_PORT}")
    print(f"   Share       : {config.GRADIO_SHARE}\n")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        show_error=config.IS_DEVELOPMENT,
        favicon_path=None,
    )
