---
title: MediScan AI
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
---

<div align="center">

# 🫀 MediScan AI
### Intelligent Medical Report Analyzer

**AI-powered analysis of medical reports — lab values, findings, personalized recommendations, and downloadable PDF reports.**

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/manuagr03/medical)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.36+-FF7C00?style=for-the-badge)](https://gradio.app)
[![OpenAI Agents](https://img.shields.io/badge/OpenAI_Agents_SDK-0.0.19+-412991?style=for-the-badge)](https://github.com/openai/openai-agents-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-27AE60?style=for-the-badge)](LICENSE)

---

</div>

## 🌐 Live Demo

**Try it now →** [https://huggingface.co/spaces/manuagr03/medical](https://huggingface.co/spaces/manuagr03/medical)

> ⚠️ The live demo uses a shared free-tier GitHub Models API key.
> Each session is limited to **2 analyses** with a **60-second cooldown** between runs.
> For unlimited use, deploy your own instance with your own GitHub API key (see [Local Setup](#-local-setup)).

---

## 📋 Table of Contents

- [What is MediScan AI?](#-what-is-mediscan-ai)
- [Who is it for?](#-who-is-it-for)
- [What it does](#-what-it-does)
- [How to use it](#-how-to-use-it)
- [Architecture](#-architecture)
- [Local Setup](#-local-setup)
- [HuggingFace Spaces Deployment](#-huggingface-spaces-deployment)
- [Project Structure](#-project-structure)
- [Build Plan (Week by Week)](#-build-plan-week-by-week)
- [API & Model Details](#-api--model-details)
- [Limitations & Disclaimer](#-limitations--disclaimer)

---

## 🩺 What is MediScan AI?

MediScan AI is an intelligent medical report analyzer built with the **OpenAI Agents SDK** and **GitHub Models**. It takes your uploaded medical document — a blood test, lab report, prescription, or discharge summary — and produces:

- A structured extraction of every lab value, medication, and clinical finding
- Color-coded abnormal flags ranked by severity
- Personalized dietary and lifestyle recommendations tied to your specific results
- An urgency assessment (Routine / Consult Soon / Urgent / Seek Immediate Care)
- A downloadable PDF report you can share with your doctor

MediScan AI does **not** replace medical advice. It helps patients understand their reports before they see their doctor, so they can have more informed conversations.

---

## 👥 Who is it for?

| User | How they use it |
|------|----------------|
| **Patients** | Upload a lab report and understand what their values mean in plain language |
| **Caregivers** | Help family members interpret test results and know when to seek care |
| **Health-conscious individuals** | Track trends in their annual bloodwork |
| **Medical students** | Practice reading and interpreting clinical reports |
| **Developers** | Study a production-grade multi-agent system built with OpenAI Agents SDK |

---

## ✨ What it does

### 📄 Step 1 — Document Parsing
- Accepts **PDF** and **DOCX/DOC** files up to 10 MB
- Block-based PDF extraction preserves table structure (solves the column-bleed problem in lab reports)
- Handles multi-page reports via intelligent text chunking

### 🔬 Step 2 — AI Report Analysis
- Classifies report type: lab report, clinical note, prescription, discharge summary
- Extracts every lab value with its result, reference range, and flag (Normal / Low / High / Borderline / Critical)
- Identifies the patient name, age, gender, report date, and ordering physician
- Detects all abnormal findings and ranks them by severity

### 💡 Step 3 — Personalized Recommendations
- Generates dietary recommendations tied to specific abnormal values
- Suggests lifestyle modifications (exercise, sleep, hydration) linked to findings
- Provides follow-up action plan with timeframes and specialist referrals
- Assigns overall urgency level based on the most critical finding

### 📋 Step 4 — Executive Summary
- Three-section summary: key findings, recommendations overview, urgency and next steps
- Processing time displayed in the status bar
- Session history panel showing all analyses in the current session

### 📥 PDF Download
- Professional PDF report with cover page, lab values table, abnormal findings, and recommendations
- Medical blue + white design with Helvetica typography
- Page numbers, header bar, and medical disclaimer on every page

---

## 🖥️ How to use it

### On the Live Demo

1. **Open** [https://huggingface.co/spaces/manuagr03/medical](https://huggingface.co/spaces/manuagr03/medical)
2. **Upload** your medical report (PDF or DOCX, max 10 MB)
3. **Click** "🔬 Analyze Report" and watch the 4-step pipeline run in real time
4. **Read** your results across 4 tabs: Findings, Recommendations, Summary, Raw Text
5. **Download** your PDF report by clicking "📥 Download PDF Report" then clicking the filename

> **Rate limits on the demo:** 2 analyses per session, 60-second cooldown between runs.
> Refresh the page to start a new session.

### Supported Report Types
- 🧪 Lab Reports — CBC, metabolic panel, lipid panel, thyroid, liver function, urine analysis
- 🩺 Clinical Notes — doctor's observations and diagnoses
- 💊 Prescriptions — medication lists with dosages
- 🏥 Discharge Summaries — hospital discharge documentation

### Tips for best results
- Use the original PDF from your lab or hospital — scanned images will have reduced accuracy
- Reports in English work best (multilingual support coming in RC3)
- Larger reports (8+ pages) take 30-60 seconds to analyze

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GRADIO UI LAYER                       │
│    File Upload  │  4 Output Tabs  │  PDF Download        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              ORCHESTRATOR (pipeline/orchestrator.py)     │
│  - Session rate limiting (2/session, 60s cooldown)       │
│  - MD5 file hash caching (session-scoped)                │
│  - Async generator — streams status updates to UI        │
│  - Recoverable pipeline — partial failure → clean error  │
└──────┬───────────────┬───────────────┬───────────────────┘
       │               │               │
┌──────▼─────┐  ┌──────▼─────┐  ┌─────▼──────────────────┐
│  TOOL 1    │  │  TOOL 2    │  │  TOOL 3                 │
│  Document  │  │  Report    │  │  Recommendation         │
│  Parser    │  │  Analyzer  │  │  Generator              │
│            │  │            │  │                         │
│ PDF→text   │  │ LLM call   │  │ LLM call                │
│ DOCX→text  │  │ Extract:   │  │ Generate:               │
│ Sanitize   │  │ findings,  │  │ diet, lifestyle,        │
│ Validate   │  │ flags,     │  │ follow-up actions,      │
│            │  │ report type│  │ urgency level           │
└────────────┘  └────────────┘  └─────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              OUTPUT BUILDER (output/pdf_builder.py)      │
│  ReportLab PDF — cover page, lab table, recommendations  │
└─────────────────────────────────────────────────────────┘
```

**Model:** `openai/gpt-4.1-mini` via GitHub Models API
**SDK:** OpenAI Agents SDK (`openai-agents>=0.0.19`)
**Temperature:** 0.1 for extraction (analyzer), 0.4 for generation (recommender)

---

## 💻 Local Setup

### Prerequisites
- Python 3.12+
- `uv` package manager (recommended) or `pip`
- A GitHub Personal Access Token with GitHub Models access

### Step 1 — Get a GitHub API Key

1. Go to [github.com](https://github.com) → **Settings** → **Developer settings**
2. Click **Personal access tokens** → **Fine-grained tokens** → **Generate new token**
3. Give it any name (e.g. `mediscan-ai`)
4. No special scopes are needed — GitHub Models access is free for all PAT holders
5. Copy the token

### Step 2 — Clone and install

```bash
# Clone the repository
git clone https://github.com/manuagr03/mediscan-ai.git
cd mediscan-ai

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

### Step 3 — Configure environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your GitHub API key
# Open .env in your editor and set:
GITHUB_API_KEY=your_github_pat_here
```

### Step 4 — Verify connectivity

```bash
# Run the connectivity test (no PDF needed)
uv run python test_connectivity.py
```

Expected output:
```
✅ ALL CHECKS PASSED — Week 1 complete!
🚀 Ready to analyze reports
🤖 Model  : openai/gpt-4.1-mini via GitHub Models
```

### Step 5 — Run the app

```bash
uv run python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Running the test suite

```bash
# Week 3 — Analyzer agent
uv run python test_analyzer.py

# Week 4 — Recommendation agent
uv run python test_recommendations.py

# Week 5 — Full orchestrator pipeline (add a PDF first)
mkdir tests
cp your_report.pdf tests/sample_report.pdf
uv run python test_orchestrator.py
```

---

## 🚀 HuggingFace Spaces Deployment

### Option A — One-Click Deploy

[![Deploy to Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-to-spaces-lg.svg)](https://huggingface.co/new-space?name=medical&sdk=gradio&template=manuagr03/medical)

Click the button above. HuggingFace will fork this Space into your account.
Then follow Step 3 below to add your secret.

### Option B — Manual Deployment (step by step)

#### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Owner:** your HuggingFace username
   - **Space name:** `medical` (or any name you prefer)
   - **License:** MIT
   - **SDK:** Gradio
   - **Visibility:** Public
3. Click **Create Space**

#### Step 2 — Push the code

```bash
# In your local project directory:

# Add HuggingFace as a remote
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/medical

# Push all files
git push hf main
```

If you don't have git set up locally:

```bash
# Install git-lfs (required by HuggingFace)
git lfs install

# Initialize git in your project if not already done
git init
git add .
git commit -m "Initial commit — MediScan AI RC1"

# Add the remote and push
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/medical
git push hf main
```

#### Step 3 — Add your GitHub API Key as a Secret

> **Critical:** Never put your API key in the code or commit it to git.
> HuggingFace Spaces provides a secure secrets vault.

1. Go to your Space page on HuggingFace
2. Click **Settings** tab
3. Scroll to **Repository secrets**
4. Click **New secret**
5. Set:
   - **Name:** `GITHUB_API_KEY`
   - **Value:** your GitHub Personal Access Token
6. Click **Save**

Your Space will automatically restart and pick up the secret as an environment variable.

#### Step 4 — Verify the deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_HF_USERNAME/medical`
2. Wait for the build to complete (2-5 minutes on first deploy)
3. The app should load and show the MediScan AI interface
4. Upload a test PDF and confirm the analysis runs

#### Troubleshooting deployment

| Problem | Fix |
|---------|-----|
| Build fails with `ModuleNotFoundError` | Check `requirements.txt` is in the root directory |
| `GITHUB_API_KEY` not found | Verify the secret name matches exactly (case-sensitive) |
| Space sleeps after 48h | Free tier Spaces sleep on inactivity — first request after sleep takes ~30s to wake |
| `app_file` error | Confirm `app.py` is in the root directory and the README frontmatter has `app_file: app.py` |
| Analysis times out | GitHub Models free tier has rate limits — wait 60 seconds and retry |

---

## 📁 Project Structure

```
mediscan-ai/
│
├── app.py                        # Gradio UI — entry point
│
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py           # Pipeline coordinator — rate limit, cache, async generator
│
├── tools/
│   ├── __init__.py
│   ├── document_parser.py        # Tool 1: PDF/DOCX → clean text
│   ├── report_analyzer.py        # Tool 2: LLM extracts findings → ReportFindings
│   └── recommendation_generator.py  # Tool 3: LLM generates advice → ReportRecommendations
│
├── output/
│   ├── __init__.py
│   └── pdf_builder.py            # ReportLab PDF generation
│
├── models/
│   ├── __init__.py
│   └── models.py                 # GitHub Models client setup
│
├── prompts/
│   ├── analyzer_prompt.py        # System prompt + user message builder for analyzer
│   └── recommendation_prompt.py  # System prompt + user message builder for recommender
│
├── utils/
│   ├── __init__.py
│   ├── validator.py              # File type, size, readability validation
│   └── sanitizer.py             # Text cleaning and chunking
│
├── custom_data_types.py          # All Pydantic models (ReportFindings, ReportRecommendations)
├── config.py                     # App configuration and environment variables
│
├── tests/
│   └── sample_report.pdf         # Sample PDF for testing (add your own)
│
├── test_connectivity.py          # Week 1 — GitHub Models connection test
├── test_parser.py                # Week 2 — Document parser test
├── test_analyzer.py              # Week 3 — Report analyzer agent test
├── test_recommendations.py       # Week 4 — Recommendation agent test
├── test_orchestrator.py          # Week 5 — Full pipeline end-to-end test
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 📅 Build Plan (Week by Week)

This project was built incrementally over 6 weeks, one layer at a time.

| Week | What was built | Key files |
|------|---------------|-----------|
| **1** | GitHub Models connectivity, config, environment setup | `config.py`, `models/models.py`, `test_connectivity.py` |
| **2** | Document parser — block-based PDF extraction, DOCX, validator, sanitizer | `tools/document_parser.py`, `utils/` |
| **3** | Report analyzer agent — LLM extraction into structured `ReportFindings` | `tools/report_analyzer.py`, `prompts/analyzer_prompt.py` |
| **4** | Recommendation agent — LLM advice into `ReportRecommendations` | `tools/recommendation_generator.py`, `prompts/recommendation_prompt.py` |
| **5** | Orchestrator, PDF builder, session state, rate limiting, caching | `pipeline/orchestrator.py`, `output/pdf_builder.py` |
| **6** | README, HuggingFace deployment, end-to-end QA | `README.md` |

**Coming in RC2:**
- 🔍 OCR support for scanned PDFs (pytesseract + pdf2image)
- 💬 Chatbot overlay for follow-up questions about the report (mem0)
- 🧠 Smarter multi-chunk merge with deduplication
- ⚡ Session-level result caching improvements
- 🌍 Drag-and-drop file upload

**Coming in RC3:**
- 📱 Progressive Web App (PWA) for mobile
- 🐳 Docker deployment
- 🌐 Multilingual support

---

## 🤖 API & Model Details

| Setting | Value |
|---------|-------|
| **Model** | `openai/gpt-4.1-mini` |
| **Provider** | GitHub Models (free tier) |
| **Base URL** | `https://models.github.ai/inference` |
| **SDK** | `openai-agents>=0.0.19` |
| **Analyzer temperature** | 0.1 (near-deterministic extraction) |
| **Recommender temperature** | 0.4 (natural language generation) |
| **Max chars per chunk** | 12,000 (fits within token budget) |
| **Cost** | Free with a GitHub account |

### Getting a GitHub Models API key

GitHub Models is **free** for all GitHub users. You don't need a credit card.

1. Sign in to [github.com](https://github.com)
2. Go to **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens**
3. Click **Generate new token**
4. Give it a name — no special permissions needed
5. Copy the token and add it to your `.env` file as `GITHUB_API_KEY`

---

## ⚠️ Limitations & Disclaimer

### Medical Disclaimer

> **MediScan AI is NOT a substitute for professional medical advice.**
>
> This tool is designed for informational and educational purposes only. The analysis generated by MediScan AI:
> - Does **not** constitute a medical diagnosis
> - Does **not** constitute professional medical advice
> - Does **not** constitute a treatment recommendation
> - Should **not** be used as the sole basis for health decisions
>
> Always consult a qualified healthcare provider regarding your medical reports, test results, and health concerns. In case of a medical emergency, contact emergency services immediately.

### Technical Limitations

| Limitation | Status | Resolution |
|-----------|--------|------------|
| Scanned PDFs (image-only) | ⚠️ Reduced accuracy | OCR coming in RC2 |
| Non-English reports | ⚠️ English only | Multilingual support in RC3 |
| Very large reports (>50 pages) | ⚠️ May time out | Chunking optimization in RC2 |
| Free tier rate limits | ℹ️ 2/session, 60s cooldown | Use local deploy for unlimited |
| No persistent storage | ℹ️ Session only | By design — privacy protection |

### Privacy

- Uploaded documents are processed **in-session only** and never stored on disk permanently
- No document content is logged or retained after the session ends
- The app does not collect any personal data
- All processing happens via the GitHub Models API — review [GitHub's privacy policy](https://docs.github.com/en/site-policy/privacy-policies/github-privacy-statement) for API data handling

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) · [GitHub Models](https://github.com/marketplace/models) · [Gradio](https://gradio.app) · [ReportLab](https://www.reportlab.com)

**[Live Demo](https://huggingface.co/spaces/manuagr03/medical)** · **[Report an Issue](https://github.com/manuagr03/mediscan-ai/issues)**

</div>
