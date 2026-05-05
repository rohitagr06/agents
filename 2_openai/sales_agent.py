from dotenv import load_dotenv
import os
import asyncio
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────
# 1. USE gemini-2.0-flash-lite  ← Free tier friendlier model
#    Flash-lite has higher free RPM than flash
# ─────────────────────────────────────────────────────────────────
MODEL = "gemini-2.0-flash"

# ─────────────────────────────────────────────────────────────────
# 2. RETRY HELPER — exponential backoff on 429s
#    Free tier limit: 15 RPM, 1M TPM, 1500 req/day (flash-lite)
#    This prevents crashes and auto-recovers from quota hits
# ─────────────────────────────────────────────────────────────────
async def with_retry(coro_fn, max_retries=5, base_delay=30):
    """
    Retry an async coroutine with exponential backoff.
    Handles 429 Resource Exhausted from Gemini free quota.
    """
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            err = str(e).lower()
            is_quota = "429" in err or "resource exhausted" in err or "quota" in err
            if is_quota and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)  # 30s, 60s, 120s, 240s...
                print(f"⚠️  Quota hit (attempt {attempt+1}/{max_retries}). "
                      f"Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise  # re-raise if not quota or out of retries

# ─────────────────────────────────────────────────────────────────
# 3. THROTTLED RUNNER HELPER
#    Adds a mandatory delay between each agent call
#    to stay under 15 RPM (1 call per 4s minimum)
# ─────────────────────────────────────────────────────────────────
INTER_CALL_DELAY = 5  # seconds between agent invocations

async def run_agent_with_throttle(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
    delay_before: float = 0.0
) -> str:
    """Run a single agent call with optional pre-delay and retry logic."""

    if delay_before > 0:
        print(f"⏳ Throttling: waiting {delay_before}s before next call...")
        await asyncio.sleep(delay_before)

    collected = []
    collected_text = []

    async def _run():
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )
        ):
            if event.is_final_response():
                for part in event.content.parts:
                    #collected.append(event.content.parts[0].text)
                    if part.text:
                        collected_text.append(part.text)
                    elif part.function_call:
                        print(f" Tool Called: {part.function_call.name}")

    await with_retry(_run)
    #return collected[0] if collected else ""
    return "".join(collected_text) if collected_text else "No text response."


# ─────────────────────────────────────────────────────────────────
# 4. SUB-AGENT DEFINITIONS
#    Kept instructions SHORT → fewer tokens → less quota burn
# ─────────────────────────────────────────────────────────────────
instructions1 = (
    "You are a professional sales agent at DipsAI, which offers an AI-powered "
    "SaaS tool for SOC2 compliance and audit preparation. "
    "Write one professional, formal cold email. Be concise — max 100 words."  # ← token limit hint
)

instructions2 = (
    "You are a witty sales agent at DipsAI, which offers an AI-powered "
    "SaaS tool for SOC2 compliance and audit preparation. "
    "Write one humorous, engaging cold email. Be concise — max 100 words."
)

instructions3 = (
    "You are a direct sales agent at DipsAI, which offers an AI-powered "
    "SaaS tool for SOC2 compliance and audit preparation. "
    "Write one ultra-concise cold email. Max 60 words — no fluff."
)

sales_agent1 = Agent(
    name="Professional_Sales_Agent",
    instruction=instructions1,
    model=MODEL   # ✅ flash-lite for all sub-agents
)

sales_agent2 = Agent(
    name="Engaging_Sales_Agent",
    instruction=instructions2,
    model=MODEL
)

sales_agent3 = Agent(
    name="Concise_Sales_Agent",
    instruction=instructions3,
    model=MODEL
)

# ─────────────────────────────────────────────────────────────────
# 5. SEND EMAIL FUNCTION TOOL
# ─────────────────────────────────────────────────────────────────
def send_email(body: str) -> dict:
    """
    Send a cold sales email to prospects.

    Args:
        body: The full email body to send.

    Returns:
        A dict with 'status' key.
    """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("xyz@gmail.com")
    to_email   = To("yzq@gmail.com")
    content    = Content("text/plain", body)
    mail       = Mail(from_email, to_email, "Cold Email from DipsAI", content).get()
    response   = sg.client.mail.send.post(request_body=mail)
    print(f"📧 SendGrid status: {response.status_code}")
    return {"status": "success"}

# ─────────────────────────────────────────────────────────────────
# 6. SALES MANAGER
#    tools[] wired up correctly with AgentTool + FunctionTool
# ─────────────────────────────────────────────────────────────────
agent1_tool = AgentTool(agent=sales_agent1)
agent2_tool = AgentTool(agent=sales_agent2)
agent3_tool = AgentTool(agent=sales_agent3)
email_tool  = FunctionTool(func=send_email)

manager_instructions = """
You are the Sales Manager at DipsAI.

Steps (follow exactly):
1. Call Professional_Sales_Agent first with the user brief → Wait for the result and get draft 1.
2. Then Call Engaging_Sales_Agent with the user brief → Wait for the result and get draft 2.
3. THEN Call Concise_Sales_Agent with the user brief → Wait for the result and get draft 3.
4. Pick the single best draft. State which one and why in one sentence.
5. Call send_email ONCE with the chosen draft body.

Rules:
- Never write drafts yourself.
- Call send_email exactly once.
- Keep your own responses SHORT — you are an orchestrator, not a writer.
"""

# ─────────────────────────────────────────────────────────────────
# 7. OPTION A: Let sales_manager orchestrate via tools (ADK native)
#    Use this if you want full ADK agent-as-tool orchestration.
#    The manager calls sub-agents sequentially by itself.
# ─────────────────────────────────────────────────────────────────
sales_manager = Agent(
    name="sales_manager",
    instruction=manager_instructions,
    model=MODEL,
    tools=[email_tool]
)

# ─────────────────────────────────────────────────────────────────
# 8. OPTION B: Manual sequential orchestration (MORE quota-safe)
#    You control the delays explicitly between each sub-agent call.
#    Uncomment this block and comment out the runner.run_async
#    in main() if Option A keeps hitting quota.
# ─────────────────────────────────────────────────────────────────
async def manual_orchestration(session_service, user_id, session_id, brief):
    """
    Manually call sub-agents one-by-one with throttling.
    More quota-safe than letting the manager fire all three rapidly.
    """
    runners = {
        "professional": Runner(agent=sales_agent1, app_name=APP_NAME, session_service=session_service),
        "engaging"     : Runner(agent=sales_agent2, app_name=APP_NAME, session_service=session_service),
        "concise"      : Runner(agent=sales_agent3, app_name=APP_NAME, session_service=session_service),
    }

    drafts = {}
    delay  = 0.0

    for name, runner in runners.items():
        print(f"\n🤖 Calling {name} agent...")
        draft = await run_agent_with_throttle(
            runner, user_id, session_id, brief, delay_before=delay
        )
        drafts[name] = draft
        print(f"✅ {name} draft received ({len(draft)} chars)")
        delay = INTER_CALL_DELAY  # apply delay for all calls after first

    return drafts


# ─────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────
APP_NAME = "sales_app"
USER_ID  = "user_2"

async def main():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID
    )

    brief = "Write a cold sales email addressed to 'Dear CEO'"

    # ── OPTION A: ADK native orchestration via sales_manager ──────
    print("\n🚀 Starting sales_manager orchestration...\n")
    manager_runner = Runner(
        agent=sales_manager,
        app_name=APP_NAME,
        session_service=session_service
    )

    result = await run_agent_with_throttle(
        runner=manager_runner,
        user_id=USER_ID,
        session_id=session.id,
        message=brief,
        delay_before=0.0
    )
    print(f"\n✅ Final manager output:\n{result}")

    # #── OPTION B: Manual sequential (uncomment if quota issues persist) ──
    # print("\n🚀 Running manual sequential orchestration...\n")
    # drafts = await manual_orchestration(
    #     session_service, USER_ID, session.id, brief
    # )
    # # Let manager pick the best — pass all drafts to it
    # combined = "\n\n---\n\n".join(
    #     f"[{k.upper()} DRAFT]\n{v}" for k, v in drafts.items()
    # )
    # pick_prompt = (
    #     f"Here are 3 email drafts:\n\n{combined}\n\n"
    #     f"Pick the best one and call send_email with its body."
    # )
    # manager_runner = Runner(
    #     agent=sales_manager,
    #     app_name=APP_NAME,
    #     session_service=session_service
    # )
    # result = await run_agent_with_throttle(
    #     manager_runner, USER_ID, session.id, pick_prompt
    # )
    # print(f"\n✅ Manager picked and sent:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())