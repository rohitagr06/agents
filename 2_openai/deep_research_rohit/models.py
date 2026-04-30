from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
from agents import OpenAIChatCompletionsModel

load_dotenv(override=True)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GITHUB_BASE_URL = "https://models.github.ai/inference"

gemini_client = AsyncOpenAI(
    base_url=GEMINI_BASE_URL,
    api_key=os.getenv("GOOGLE_API_KEY")
)
github_client = AsyncOpenAI(
    base_url=GITHUB_BASE_URL,
    api_key=os.getenv("GITHUB_API_KEY")
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=gemini_client
)
github_model = OpenAIChatCompletionsModel(
    model="openai/gpt-4.1-mini",
    openai_client=github_client
)