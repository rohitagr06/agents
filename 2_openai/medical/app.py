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

import logging
import gradio as gr
from datetime import datetime
import config
from pipeline.orchestrator import MediScanOrchestrator, AnalysisResult, SessionState
from output.pdf_builder import generate_pdf
from html import escape

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
  MediScan AI &nbsp;·&nbsp; RC1 &nbsp;·&nbsp; Built with AI Models + OpenAI Agents SDK + Gradio
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

logger = logging.getLogger("app")

# Module-level orchestrator instance — stateless, reusable across requests
_orchestrator = MediScanOrchestrator()

# ─────────────────────────────────────────────────────────────
#  analyze_report()  — main Gradio event handler
#
#  This is an async GENERATOR function — it yields intermediate
#  status updates so Gradio streams progress to the UI in real time.
#
#  HOW GRADIO STREAMING WORKS:
#  When a Gradio event handler is an async generator (uses yield),
#  Gradio streams each yielded tuple to the UI immediately.
#  This is how we show "📄 Parsing..." → "🔬 Analyzing..." live
#  without the UI freezing during the 20-30 second analysis.
#
#  YIELD SIGNATURE (must match outputs= list in .click()):
#  (findings_md, recommendations_md, summary_md, raw_md,
#   status, session_state, download_btn_update, history_update)
# ─────────────────────────────────────────────────────────────


async def analyze_report(file, session_state: dict):
    """
    Async generator — streams status updates then final result to Gradio.

    Args:
        file          : Gradio file object from gr.File
        session_state : dict representation of SessionState from gr.State()
    """
    # Rehydrate SessionState from gr.State dict
    # (gr.State serialises to dict — we reconstruct the dataclass)
    state = SessionState(
        analyses_used=session_state.get("analyses_used", 0),
        last_analysis_time=session_state.get("last_analysis_time", 0.0),
        cache=session_state.get("_cache_obj", {}),
    )

    # Empty result for streaming intermediate status updates
    def _status_yield(msg: str):
        return (
            "",
            "",
            "",
            "",
            msg,
            session_state,
            gr.update(visible=False),
            gr.update(),
        )

    async for update in _orchestrator.run(file, state):

        if isinstance(update, str):
            # Intermediate status message — stream to status_box
            yield _status_yield(update)

        elif isinstance(update, SessionState):
            # Orchestrator finished — save updated state back to gr.State
            session_state = {
                "analyses_used": update.analyses_used,
                "last_analysis_time": update.last_analysis_time,
                "_cache_obj": update.cache,
            }

        elif isinstance(update, AnalysisResult):
            result = update

            if not result.success:
                yield (
                    "",
                    "",
                    "",
                    "",
                    result.status,
                    session_state,
                    gr.update(visible=False),
                    gr.update(),
                )
                return

            # Store findings + recs in session_state for PDF download
            session_state["_last_findings"] = result.findings
            session_state["_last_recs"] = result.recommendations

            # Append to session history so the panel shows past analyses
            session_state.setdefault("_history", []).append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "filename": (
                        file.name.split("/")[-1] if hasattr(file, "name") else "Unknown"
                    ),
                    "urgency": (
                        result.recommendations.overall_urgency
                        if result.recommendations
                        else "routine"
                    ),
                }
            )

            # Build history entry
            history_html = _build_history_html(session_state)

            yield (
                result.findings_md,
                result.recommendations_md,
                result.summary_md,
                result.raw_md,
                result.status,
                session_state,
                gr.update(visible=True),  # show Download PDF button
                gr.update(value=history_html),
            )


def download_pdf(session_state: dict):
    """
    Generates PDF and returns the filepath.
    gr.File serves it — user clicks the filename once to download.
    This is the most reliable approach across all Gradio versions.

    WHY NOT gr.HTML WITH <script>:
    Gradio sanitizes injected scripts in gr.HTML for security — the
    script tag never executes, so auto-click approaches don't work.

    WHY NOT gr.DownloadButton:
    Inconsistent behaviour across Gradio 4.x minor versions.

    RESULT: one click on Download PDF button → file widget appears →
    one click on filename → browser downloads the PDF.
    """
    findings = session_state.get("_last_findings")
    recs = session_state.get("_last_recs")

    if findings is None or recs is None:
        logger.warning("Download PDF clicked but no analysis in session.")
        return gr.update(visible=False)

    try:
        pdf_path = generate_pdf(findings, recs)
        return gr.update(value=pdf_path, visible=True)
    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        return gr.update(visible=False)


async def clear_all():
    """Reset all outputs and session state."""
    empty_state = {
        "analyses_used": 0,
        "last_analysis_time": 0.0,
        "_cache_obj": {},
        "_last_findings": None,
        "_last_recs": None,
    }
    return (
        None,
        "",
        "",
        "",
        "",
        "Ready. Upload a report to begin.",
        empty_state,
        gr.update(visible=False),
        gr.update(value=_HISTORY_PLACEHOLDER),
    )


# ─────────────────────────────────────────────────────────────
#  History Panel Helpers
# ─────────────────────────────────────────────────────────────

_HISTORY_PLACEHOLDER = "<p style='color:#5A7A96; font-size:0.82rem; padding:8px;'>No analyses yet this session.</p>"


def _build_history_html(session_state: dict) -> str:
    """Build the HTML for the collapsible history panel."""
    entries = session_state.get("_history", [])
    if not entries:
        return _HISTORY_PLACEHOLDER

    rows = ""
    for entry in reversed(entries):  # newest first
        urgency_icon = {
            "routine": "✅",
            "consult_soon": "🟡",
            "urgent": "🔴",
            "seek_immediate_care": "🚨",
        }.get(entry.get("urgency", "routine"), "⚪")
        safe_time = escape(str(entry.get("time", "")))
        safe_filename = escape(str(entry.get("filename", "")))
        rows += (
            f"<div style='padding:6px 8px; border-bottom:1px solid rgba(0,196,180,0.1);'>"
            f"<div style='font-size:0.8rem; color:#A8C0D6;'>{safe_time}</div>"
            f"<div style='font-size:0.85rem; color:#F0F6FF; margin-top:2px;'>"
            f"{urgency_icon} {safe_filename}</div>"
            f"</div>"
        )
    return rows or _HISTORY_PLACEHOLDER


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

        # ── Hidden session state ──────────────────────────────
        # gr.State() persists per browser session — cleared on refresh.
        # Stores: analyses_used, last_analysis_time, cache, last findings/recs.
        session_state = gr.State(
            {
                "analyses_used": 0,
                "last_analysis_time": 0.0,
                "_cache_obj": {},
                "_last_findings": None,
                "_last_recs": None,
                "_history": [],
            }
        )

        # ── Header ──
        gr.HTML(HEADER_HTML)
        gr.HTML(DISCLAIMER_HTML)
        gr.HTML(HOW_IT_WORKS_HTML)

        # ── Main Layout ──
        with gr.Row(equal_height=False):

            # ── LEFT COLUMN — Upload & Controls ──────────────
            with gr.Column(scale=4, min_width=300):

                gr.HTML('<div class="section-label">Upload Document</div>')

                file_input = gr.File(
                    label="",
                    file_types=[".pdf", ".docx", ".doc"],
                    file_count="single",
                    elem_classes=["upload-zone"],
                    height=170,
                )

                gr.HTML(
                    """
                <div style="font-size:0.75rem; color:#5A7A96; text-align:center; margin: -8px 0 14px; line-height:1.5;">
                  Supported formats: PDF · DOCX · DOC &nbsp;|&nbsp; Max size: 10 MB<br>
                  Your document is processed in-session and never stored.
                </div>
                """
                )

                analyze_btn = gr.Button(
                    "🔬  Analyze Report",
                    variant="primary",
                    elem_classes=["analyze-btn"],
                )

                gr.HTML('<div style="height:8px;"></div>')

                # ── Download PDF button (hidden until analysis complete) ──
                download_btn = gr.Button(
                    "📥  Download PDF Report",
                    variant="secondary",
                    elem_classes=["download-btn"],
                    visible=False,
                )

                # gr.HTML receives the self-clicking download anchor from download_pdf()
                pdf_output = gr.File(
                    label="📄 PDF Ready — click filename to download",
                    visible=False,
                    elem_classes=["pdf-output"],
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
                    interactive=False,
                    lines=3,
                    elem_classes=["status-box"],
                    container=False,
                )

                # ── Collapsible History Panel ─────────────────
                with gr.Accordion("🕐 Session History", open=False):
                    history_panel = gr.HTML(
                        value=_HISTORY_PLACEHOLDER,
                        elem_id="history-panel",
                    )

                # ── Info Box ──────────────────────────────────
                gr.HTML(
                    """
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
                """
                )

                gr.HTML(
                    """
                <div style="margin-top: 12px; background: rgba(79,195,247,0.04);
                     border: 1px solid rgba(79,195,247,0.12); border-radius: 10px; padding: 14px;">
                  <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                       color:#4FC3F7; font-family:'DM Mono',monospace; margin-bottom:8px;">
                    Powered By
                  </div>
                  <div style="font-size:0.78rem; color:#A8C0D6; line-height:1.9;">
                    🤖 openai/gpt-4.1-mini<br>
                    ⚡ GitHub Models API<br>
                    🔗 OpenAI Agents SDK
                  </div>
                </div>
                """
                )

            # ── RIGHT COLUMN — Output ─────────────────────────
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

        # ── Event Handlers ────────────────────────────────────

        # Analyze — async generator, streams 8-tuple on each yield
        analyze_btn.click(
            fn=analyze_report,
            inputs=[file_input, session_state],
            outputs=[
                findings_output,  # 1
                recommendations_output,  # 2
                summary_output,  # 3
                raw_output,  # 4
                status_box,  # 5
                session_state,  # 6 — updated rate limit + cache
                download_btn,  # 7 — visible=True on success
                history_panel,  # 8 — updated HTML
            ],
            show_progress="full",
        )

        # Download PDF — returns self-clicking HTML anchor, no .then() needed
        download_btn.click(
            fn=download_pdf,
            inputs=[session_state],
            outputs=[pdf_output],
        )

        # Clear — resets everything including session state
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
                session_state,
                download_btn,
                history_panel,
            ],
        )

    return app


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)

    is_valid, errors = config.validate_config()
    if not is_valid:
        print("\n❌ Cannot start — missing required config:")
        for err in errors:
            print(f"   → {err}")
        print("\n💡 Fix: cp .env.example .env  then add your GITHUB_API_KEY\n")
        raise SystemExit(1)

    print(f"\n🫀 Starting {config.APP_TITLE} {config.APP_VERSION}")
    print("   Model       : openai/gpt-4.1-mini via GitHub Models")
    print("   SDK         : openai-agents (Agent + Runner.run)")
    print(f"   Environment : {config.APP_ENV}")
    print(f"   Port        : {config.GRADIO_SERVER_PORT}")
    print(f"   Share       : {config.GRADIO_SHARE}")
    print(f"   Rate limit  : {2} analyses/session · 60s cooldown\n")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=config.GRADIO_SERVER_PORT,
        share=config.GRADIO_SHARE,
        show_error=config.IS_DEVELOPMENT,
        favicon_path=None,
    )
