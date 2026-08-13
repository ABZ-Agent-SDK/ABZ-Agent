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
        model="gemini-2.5-flash",
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


class TestIterativeModeGuaranteesAFinalAnswer:
    """Real bug: a model that spends every iteration on a tool call (e.g. it
    keeps calling a misspelled/nonexistent tool name and never converges)
    exhausted the entire max_iterations budget with none left over to
    actually answer, producing the unhelpful "Reached iteration limit
    without final answer" fallback — even though budget nominally looked
    sufficient for "one tool call then answer".

    Root cause: Agent._run()'s iterative loop offered tools on every single
    iteration, including the last one. If the model used its last iteration
    on yet another tool call, there was no turn left to produce text.

    Fix: the last iteration of the loop no longer offers tools at all
    (Agent._build_prompt(..., allow_tools=False) /
    Agent._generate_and_dispatch(..., allow_tools=False)). For native
    tool-calling providers (Gemini, Groq — everything this SDK ships) this
    makes it structurally impossible for that call to request a tool, so it
    must return plain text — guaranteeing a real final answer for any
    max_iterations >= 2 setup, regardless of how many earlier turns were
    wasted on tool calls (wrong name, repeated, or otherwise)."""

    def test_model_stuck_calling_wrong_tool_name_still_gets_a_final_answer(self, monkeypatch):
        """Minimal reproduction of the reported bug: a fake tool ("research")
        that does not match any registered tool name, called repeatedly. No
        Tavily import, no API key required."""
        def tavily_search(query: str) -> dict:
            """Search the web. Args: query: the search query"""
            return TAVILY_SHAPED_RESULT  # never actually reached — name never matches

        agent = Agent(
            name="Researcher",
            instructions=(
                "Use the search tool to answer the user's question. "
                "After using the tool, summarize the findings in plain, friendly language."
            ),
            model="gemini-2.5-flash",
            tools=[tavily_search],
            max_iterations=3,
        )

        # The model calls a tool name that was never registered, on every
        # turn it's offered one — exactly the reported failure mode.
        fake, seen_prompts = _scripted_provider([
            ToolCall(tool="research", args={"query": "Karachi weather forecast"}),
            ToolCall(tool="research", args={"query": "Karachi weather forecast"}),
            "I wasn't able to look up live weather data, but here's what I can tell you generally.",
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("What's the weather in Karachi today?")

        # A real, final natural-language answer — not the iteration-limit fallback.
        assert result.content == "I wasn't able to look up live weather data, but here's what I can tell you generally."
        assert "Reached iteration limit" not in result.content
        assert len(seen_prompts) == 3  # 2 tool-call attempts + 1 forced-final answer turn

    def test_last_iteration_offers_no_tools(self, monkeypatch):
        """Direct check on the mechanism: the final call's prompt has no tool
        manifest (fallback-mode framing) and no tools are passed natively."""
        def broken_tool(query: str) -> str:
            """Always the wrong tool. Args: query: anything"""
            return "n/a"

        agent = Agent(
            name="Researcher",
            instructions="Use the tool, then answer.",
            model="gemini-2.5-flash",
            tools=[broken_tool],
            max_iterations=2,  # smallest possible: 1 tool-call turn + 1 forced-final turn
        )

        captured_tools = []

        def fake_generate(self, prompt, *, tools=None, output_schema=None, strict=True):
            captured_tools.append(tools)
            if len(captured_tools) == 1:
                return GenerationResult(tool_calls=[ToolCall(tool="broken_tool", args={"query": "x"})])
            return GenerationResult(text="Final answer without another tool call.")

        monkeypatch.setattr(GeminiProvider, "generate", fake_generate)

        result = agent.run("hi")

        assert result.content == "Final answer without another tool call."
        assert captured_tools[0] is not None       # first (non-final) call: tools offered
        assert captured_tools[1] is None            # last (forced-final) call: no tools offered

    def test_well_behaved_single_tool_call_workflow_unaffected(self, monkeypatch):
        """Regression guard: a model that behaves correctly (one tool call,
        then answers) must still work exactly as before — the fix only
        changes behavior on the last iteration, which a well-behaved run
        never reaches."""
        def tavily_search(query: str) -> dict:
            """Search the web. Args: query: the search query"""
            return TAVILY_SHAPED_RESULT

        agent = Agent(
            name="Researcher",
            instructions="Search, then summarize.",
            model="gemini-2.5-flash",
            tools=[tavily_search],
            max_iterations=3,
        )
        fake, seen_prompts = _scripted_provider([
            ToolCall(tool="tavily_search", args={"query": "Karachi weather today"}),
            "It's 34°C and partly cloudy in Karachi.",
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("What's the weather in Karachi today?")
        assert result.content == "It's 34°C and partly cloudy in Karachi."
        assert len(seen_prompts) == 2  # loop returned early, third (forced-final) turn never needed

    def test_multiple_legitimate_tool_calls_still_work(self, monkeypatch):
        """Two different tools called in sequence, then a final answer, still
        completes normally when there's enough budget."""
        def search(query: str) -> dict:
            """Search. Args: query: text"""
            return {"result": "raw search data"}

        def calculate(expression: str) -> str:
            """Calculate. Args: expression: a math expression"""
            return "42"

        agent = Agent(
            name="Researcher",
            instructions="Use tools as needed, then answer.",
            model="gemini-2.5-flash",
            tools=[search, calculate],
            max_iterations=4,
        )
        fake, seen_prompts = _scripted_provider([
            ToolCall(tool="search", args={"query": "x"}),
            ToolCall(tool="calculate", args={"expression": "6*7"}),
            "Based on my research and calculation, the answer is 42.",
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("What's the answer?")
        assert result.content == "Based on my research and calculation, the answer is 42."
        assert len(seen_prompts) == 3  # 2 tool calls + final answer, well within the 4-iteration budget

    def test_agent_without_tools_unaffected(self, monkeypatch):
        """Regression guard: a tool-less agent's iterative loop (rare but
        valid — e.g. for structured-output retries) is completely untouched
        by this fix, since allow_tools=False only ever matters when
        self.tools is non-empty."""
        agent = Agent(
            name="Chatty",
            instructions="Just chat.",
            model="gemini-2.5-flash",
            max_iterations=3,
        )
        fake, seen_prompts = _scripted_provider(["Hello there!"])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("hi")
        assert result.content == "Hello there!"
        assert len(seen_prompts) == 1

    def test_genuinely_exhausted_budget_still_falls_back_gracefully(self, monkeypatch):
        """The iteration-limit fallback message is NOT deleted — it's simply
        no longer reachable via the tool-call-starvation failure mode for
        native providers. This test forces it via a provider that returns
        completely empty text on the forced-final turn, confirming the
        safety net still exists and still returns cleanly rather than
        raising."""
        def broken_tool() -> str:
            """A tool. Args: none"""
            return "data"

        agent = Agent(
            name="Researcher",
            instructions="Use the tool, then answer.",
            model="gemini-2.5-flash",
            tools=[broken_tool],
            max_iterations=1,  # single-turn mode: separate, pre-existing code path
        )
        fake, _ = _scripted_provider([
            ToolCall(tool="broken_tool", args={}),
        ])
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        # Single-turn mode is untouched by this fix — still returns the raw
        # tool result directly, exactly as documented.
        result = agent.run("hi")
        assert result.content == "data"
