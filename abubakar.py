from abzagent import function_tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@function_tool
def tavily_search(query: str) -> str:
    """
    Search the web using Tavily SDK and return a concise summary of results.
    Args:
      query: The search query string.
    """
    # Use SDK: synchronous search
    resp = tavily_client.search(query=query, max_results=5, search_depth="basic")
    # Extract useful parts
    results = resp.get("results", [])
    if not results:
        return "No results found."

    # Build a summary string
    summary_lines = []
    for r in results[:3]:
        title = r.get("title", "<no title>")
        url = r.get("url", "<no url>")
        snippet = r.get("content", "").strip()
        summary_lines.append(f"{title} — {url}\n{snippet}")

    return "Top Tavily results:\n\n" + "\n\n".join(summary_lines)
from abzagent import Agent

agent = Agent(
    name="ResearcherAgent",
    instructions="Searches the web using Tavily",
    tools=[tavily_search],
)

res = agent.run("Search latest BMW 7 Series specs")
print(res.content)