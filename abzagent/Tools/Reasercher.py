# researcher_simple.py
from __future__ import annotations
import os
from typing import List
from dotenv import load_dotenv

# SDK
from abzagent import Agent, Memory, function_tool
# Tavily
from tavily import TavilyClient

load_dotenv()  # reads .env

# ---- tiny web search tool (Tavily) ----
def _tavily() -> TavilyClient:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Missing TAVILY_API_KEY. Put it in .env")
    return TavilyClient(api_key=key)

@function_tool
def web_search(se , query: str, max_results: int = 6) -> str:
    """
    Search the web and return a short digest with sources.

    Args:
        query: What to search for.
        max_results: How many links to include (default 6).
    """
    t = _tavily()
    r = t.search(query=query, max_results=max_results, search_depth="advanced", include_answer=True)
    bits: List[str] = []
    if r.get("answer"):
        bits.append(f"Answer: {r['answer']}")
    for i, it in enumerate(r.get("results", []), start=1):
        title = it.get("title") or "Untitled"
        url = it.get("url") or ""
        bits.append(f"{i}. {title} — {url}")
    return "\n".join(bits) if bits else "No results."

def main():
    # Make sure your Gemini key is available
    gemini = os.getenv("GEMINI_API_KEY")
    if not gemini:
        print("Missing GEMINI_API_KEY. Put it in .env")
        return

    agent = Agent(
        name="AI Researcher",
        instructions=(
            "You are a helpful research assistant.\n"
            "- Use the web_search tool when the question needs current info.\n"
            "- Give a short summary (3 6 bullet points) and list sources as URLs.\n"
        ),
        model="gemini-2.0-flash",
        tools=[web_search],
        memory=Memory(),
        verbose=False,       # keep output clean
        max_iterations=4,
        api_key=gemini,      # pass YOUR OWN key from .env
    )

    print("==== Simple Researcher ====")
    print("Type your question (q to quit).")
    while True:
        q = input("You > ").strip()
        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            print("Bye!")
            break
        res = agent.run(q)
        print("\n--- Answer ---")
        print(res.content)
        print("--------------\n")

if __name__ == "__main__":
    main()
