"""
models/models.py — GitHub Models client and model setup
Follows the exact same pattern from the Deep Research project.

Provider : GitHub Models (https://models.github.ai/inference)
SDK      : openai-agents (pip install openai-agents)
Token    : GitHub PAT → github.com › Settings › Developer settings › PAT › Fine-grained tokens

Usage in any agent file:
    from models.models import github_model
"""

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# ─────────────────────────────────────────────
#  GitHub Models — Client & Model
#  Exact pattern from your Deep Research project
# ─────────────────────────────────────────────

GITHUB_BASE_URL = "https://models.github.ai/inference"

github_client = AsyncOpenAI(
    base_url=GITHUB_BASE_URL,
    api_key=os.environ.get("GITHUB_API_KEY"),
)

github_model = OpenAIChatCompletionsModel(
    model="openai/gpt-4.1-mini",
    openai_client=github_client,
)
