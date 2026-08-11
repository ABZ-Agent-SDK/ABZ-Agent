"""
Regression tests for how a tool's return value flows into AgentResult.content.

Context: a real-world research tool (Tavily-shaped) returns a large raw dict.
Printing/using that raw payload as the user-facing answer looked like an SDK
bug at first, but tracing Agent._run() (core/agent.py) shows this is the
already-documented behavior of single-turn mode (max_iterations <= 1, the
default): a tool call's result becomes AgentResult.content directly, with no
follow-up model call to turn it into a natural-language answer (see
"Single-turn vs. multi-step mode" in docs/agents.md). Iterative mode
(max_iterations > 1) is the existing, correct mechanism for "the agent sees
the raw tool result as context, then produces a clean final answer" — these
tests lock in both halves of that contract so neither can silently regress.
"""
import json
import os
import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent import Agent
from abzagent.providers.gemini import GeminiProvider
from abzagent.providers.base import GenerationResult
from abzagent.core.tools import ToolCall


def _scripted_provider(responses):
    """responses: list of items consumed in order across successive calls.
    Each item is either a plain text string (-> GenerationResult(text=...))
    or a ToolCall (-> GenerationResult(tool_calls=[...])). Also records every
    prompt seen, so tests can assert on what context a later call received."""
    seen_prompts = []

    def fake_generate(self, prompt, *, tools=None, output_schema=None, strict=True):
        seen_prompts.append(prompt)
        item = responses.pop(0)
        if isinstance(item, ToolCall):
            return GenerationResult(tool_calls=[item])
        return GenerationResult(text=item)

    return fake_generate, seen_prompts


TAVILY_SHAPED_RESULT = {
    "query": "Karachi weather today",
    "follow_up_questions": None,
    "answer": None,
    "images": [],
    "results": [
        {"title": "Karachi Weather", "url": "https://example.com/1",
         "content": "34C, partly cloudy, humidity 65%.", "score": 0.95},
    ],
    "response_time": 1.49,
}


def _make_search_agent(**kwargs):
    def tavily_search(query: str) -> dict:
        """Search the web. Args: query: the search query"""
        return TAVILY_SHAPED_RESULT

    return Agent(
        name="Researcher",
        instructions="You are a research assistant. Use the search tool to answer questions.",
        model="gemini-2.0-flash",
        tools=[tavily_search],
        **kwargs,
    )


class TestSingleTurnModeDocumentedBehavior:
    """Locks in the existing, documented contract: in single-turn mode (the
    default), a tool call's result IS AgentResult.content — no interpretation
    step. This is intentional; these tests exist so nobody "fixes" it later
    without realizing it's the documented, relied-upon design."""

    def test_tool_result_becomes_content_directly(self, monkeypatch):
        agent = _make_search_agent()  # max_iterations unset -> single-turn
        fake, seen_prompts = _scripted_provider([
            ToolCall(tool="tavily_search", args={"query": "Karachi weather today"}),
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("What's the weather in Karachi today?")

        # The raw (stringified) tool payload IS the content — by design.
        assert result.content == str(TAVILY_SHAPED_RESULT)
        assert "follow_up_questions" in result.content  # unmistakably the raw payload

    def test_no_follow_up_model_call_after_tool_use(self, monkeypatch):
        """Single-turn really is single: one tool call, zero interpretation
        calls. If this ever becomes 2, single-turn mode's contract changed."""
        agent = _make_search_agent()
        fake, seen_prompts = _scripted_provider([
            ToolCall(tool="tavily_search", args={"query": "x"}),
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        agent.run("What's the weather?")
        assert len(seen_prompts) == 1


class TestIterativeModeInterpretsToolResult:
    """The existing, correct mechanism for the desired architecture:
    User -> Agent -> Tool -> raw result -> Agent interprets -> clean response.
    max_iterations > 1 gives the agent a follow-up turn with the raw tool
    result available as context, instead of returning it verbatim."""

    def test_final_content_is_the_agents_interpretation_not_the_raw_payload(self, monkeypatch):
        agent = _make_search_agent(max_iterations=3)
        fake, seen_prompts = _scripted_provider([
            ToolCall(tool="tavily_search", args={"query": "Karachi weather today"}),
            "It's 34°C and partly cloudy in Karachi right now, with 65% humidity.",
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("What's the weather in Karachi today?")

        assert result.content == "It's 34°C and partly cloudy in Karachi right now, with 65% humidity."
        assert "follow_up_questions" not in result.content  # raw payload did NOT leak to the user
        assert len(seen_prompts) == 2  # one tool-call turn, one interpretation turn

    def test_raw_tool_result_is_available_to_the_agent_as_context(self, monkeypatch):
        """Requirement: raw tool data must still reach the agent (just not
        the end user). The follow-up prompt must contain the tool's output —
        rendered via Memory.to_prompt() as a "[TOOL]: <result>" line (the
        actual mechanism; the tool='...' role is recorded via
        self.memory.remember("tool", obs) in Agent._run())."""
        agent = _make_search_agent(max_iterations=3)
        fake, seen_prompts = _scripted_provider([
            ToolCall(tool="tavily_search", args={"query": "Karachi weather today"}),
            "Sunny and 34 degrees in Karachi today.",
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        agent.run("What's the weather in Karachi today?")

        follow_up_prompt = seen_prompts[1]
        assert "[TOOL]:" in follow_up_prompt
        assert "follow_up_questions" in follow_up_prompt  # the raw dict, present as context
        assert "34C, partly cloudy" in follow_up_prompt

    def test_raw_tool_result_is_inspectable_via_agent_memory(self, monkeypatch):
        """result.steps in iterative mode only records the tool *call*
        (Agent._run() appends call_repr but not obs to steps on this path) —
        the raw result's audit trail lives in agent.memory instead."""
        agent = _make_search_agent(max_iterations=3)
        fake, _ = _scripted_provider([
            ToolCall(tool="tavily_search", args={"query": "x"}),
            "Clean final answer.",
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("weather?")
        remembered = [m.content for m in agent.memory.load() if m.role == "tool"]
        assert any("follow_up_questions" in c for c in remembered)  # raw tool output still inspectable
        assert result.content == "Clean final answer."


class TestToolReturnValueStringification:
    """core/tools.py's _call_maybe_async() coerces non-str tool return values
    via str(result) — documents existing, generic (non-Tavily-specific)
    behavior so a large dict/list return is never silently mishandled."""

    def test_dict_return_value_is_stringified(self, monkeypatch):
        agent = _make_search_agent()
        fake, _ = _scripted_provider([
            ToolCall(tool="tavily_search", args={"query": "x"}),
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("weather?")
        assert isinstance(result.content, str)
        # round-trips back to the same structure (Python repr of the dict)
        assert result.content == repr(TAVILY_SHAPED_RESULT)
