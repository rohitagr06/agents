from agents import Agent
from custom_data_types import WebSearchPlan
from models import github_model

HOW_MANY_SEARCHES = 10

INSTRUCTIONS = f"""
You are a helpful reserach assistant. Given a query, comeup with a set of web searches to perform to best answer the query.
Output {HOW_MANY_SEARCHES} terms to query for.
"""

planner_agent = Agent(
    name = "Planner Agent",
    instructions=INSTRUCTIONS,
    output_type=WebSearchPlan,
    model=github_model
)

