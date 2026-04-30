from tavily import TavilyClient
import os
from dotenv import load_dotenv
from agents import function_tool

load_dotenv(override=True)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@function_tool
def web_search(query: str) -> str:
    """
    Search the web for the given query and return a summarized result.
    Use this to find current information on any topic.

    Args:
        query: The search term to look up.

    Returns:
        A string containing the search results summary. 
    """

    print(f"  🔍 Searching: {query}")

    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True
        )

        #Build a clean summary from results
        answer = response.get("answer", "")
        results = response.get("results", [])

        summary = f"ANSWER: {answer}\n\nSOURCES:\n"

        for i, r in enumerate(results, 1):
            summary += f"\n[{i}] {r['title']}\n"
            summary += f"\n   URL: {r['url']}\n"
            summary += f"\n   {r['content'][:300]}...\n"

        return summary

    except Exception as e:
        return f"Search failed: {str(e)}"