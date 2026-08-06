"""
Tests for Handoffs (abzagent/core/handoffs.py + Agent._run()/_perform_handoff()).

Covers: single handoff, multiple handoffs, nested/chained handoffs, circular
handoffs, max depth, invalid targets, memory/context preservation, shared-memory
non-duplication, tool compatibility, structured-output compatibility, backward
compatibility, on_handoff/input_type, graceful bad-args, input_filter, and the
pre-existing unknown-tool error path (regression guard).

Provider mocks return GenerationResult (native tool-calling shape) rather than
bare JSON-blob strings — tests assert against the semantic ToolCall, not a
fallback-path serialization detail.
"""
import os
import pytest
from pydantic import BaseModel

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent import Agent, AgentResult
from abzagent.providers.gemini import GeminiProvider
from abzagent.providers.base import GenerationResult
from abzagent.core.tools import ToolCall
from abzagent.core.handoffs import (
    handoff,
    CircularHandoffError,
    MaxHandoffDepthExceededError,
    InvalidHandoffTargetError,
)
from abzagent.core.context import RunContextWrapper
import abzagent.core.handoffs as handoffs_mod
from abzagent.extensions.handoffs_filter import remove_all_tools, keep_last_n_turns


def _scripted_provider(responses):
    """responses: {agent_name: item_or_list_of_items}, where each item is either
    a plain text string (-> GenerationResult(text=...)) or a ToolCall (->
    GenerationResult(tool_calls=[...])). Each generate() call is routed by
    matching '[AGENT NAME]: <name>' in the prompt; list values are consumed in
    order across successive calls to the same agent."""
    calls = {"log": []}

    def fake_generate(self, prompt, *, tools=None, output_schema=None, strict=True):
        for name, resp in responses.items():
            if f"[AGENT NAME]: {name}" in prompt:
                calls["log"].append(name)
                item = resp.pop(0) if isinstance(resp, list) else resp
                if isinstance(item, ToolCall):
                    return GenerationResult(tool_calls=[item])
                return GenerationResult(text=item)
        raise AssertionError(f"No scripted response matched prompt:\n{prompt}")

    return fake_generate, calls


@pytest.fixture(autouse=True)
def _patch_provider(monkeypatch):
    # Each test sets abzagent.providers.gemini.GeminiProvider.generate itself via
    # _scripted_provider; this fixture just ensures a clean default so a test that
    # forgets to patch fails loudly instead of hanging on a real network call.
    def unset_generate(self, prompt, *, tools=None, output_schema=None, strict=True):
        raise AssertionError("Test forgot to patch GeminiProvider.generate")

    monkeypatch.setattr(GeminiProvider, "generate", unset_generate)
    yield


def make_agent(name, **kwargs):
    return Agent(name=name, instructions=f"You are {name}.", model="gemini-2.0-flash", **kwargs)


class TestSingleHandoff:
    def test_basic_handoff_routes_to_target(self, monkeypatch):
        billing = make_agent("Billing")
        support = make_agent("Support", handoffs=[billing])

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={"message": "Needs an invoice."}),
            "Billing": "Sure, I can help with your invoice.",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = support.run("I need help with my invoice.")
        assert result.content == "Sure, I can help with your invoice."
        assert result.last_agent is billing

    def test_steps_include_handoff_marker(self, monkeypatch):
        billing = make_agent("Billing")
        support = make_agent("Support", handoffs=[billing])
        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": "Done.",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = support.run("hi")
        assert any("[HANDOFF] Support -> Billing" in s for s in result.steps)

    def test_no_handoffs_configured_backward_compatible(self, monkeypatch):
        agent = make_agent("Solo")
        fake, _ = _scripted_provider({"Solo": "just a plain answer"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("hi")
        assert result.content == "just a plain answer"
        assert result.last_agent is agent


class TestMultipleHandoffs:
    def test_correct_target_chosen(self, monkeypatch):
        research = make_agent("Research")
        writer = make_agent("Writer")
        review = make_agent("Review")
        planner = make_agent("Planner", handoffs=[research, writer, review])

        fake, calls = _scripted_provider({
            "Planner": ToolCall(tool="transfer_to_writer", args={}),
            "Research": "should not be called",
            "Writer": "writer handled it",
            "Review": "should not be called",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = planner.run("write something")
        assert result.content == "writer handled it"
        assert result.last_agent is writer
        assert "Research" not in calls["log"]
        assert "Review" not in calls["log"]

    def test_matches_user_spec_shape(self, monkeypatch):
        """Exact shape from the feature request: bare Agent instances in handoffs=[...]."""
        research_agent = make_agent("Research")
        writer_agent = make_agent("Writer")
        review_agent = make_agent("Review")

        planner = Agent(
            name="Planner",
            instructions="Route tasks to the correct specialist.",
            model="gemini-2.0-flash",
            handoffs=[research_agent, writer_agent, review_agent],
        )

        fake, _ = _scripted_provider({
            "Planner": ToolCall(tool="transfer_to_research", args={"message": "Look into X."}),
            "Research": "Here is what I found about X.",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = planner.run("Research X for me.")
        assert result.last_agent.name == "Research"
        assert result.content == "Here is what I found about X."


class TestNestedHandoffs:
    def test_chained_a_to_b_to_c(self, monkeypatch):
        c = make_agent("C")
        b = make_agent("B", handoffs=[c])
        a = make_agent("A", handoffs=[b])

        fake, _ = _scripted_provider({
            "A": ToolCall(tool="transfer_to_b", args={}),
            "B": ToolCall(tool="transfer_to_c", args={}),
            "C": "final answer from C",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = a.run("start")
        assert result.last_agent is c
        assert result.content == "final answer from C"
        joined = "\n".join(result.steps)
        assert "[HANDOFF] A -> B" in joined
        assert "[HANDOFF] B -> C" in joined
        assert joined.index("[HANDOFF] A -> B") < joined.index("[HANDOFF] B -> C")


class TestCircularAndDepth:
    def test_self_handoff_raises(self, monkeypatch):
        a = make_agent("A")
        a.tools[handoff(a).to_tool(a).name] = handoff(a).to_tool(a)
        a._handoffs.append(handoff(a))

        fake, _ = _scripted_provider({"A": ToolCall(tool="transfer_to_a", args={})})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(CircularHandoffError):
            a.run("start")

    def test_circular_a_b_a_raises(self, monkeypatch):
        a = make_agent("A")
        b = make_agent("B", handoffs=[a])
        a_handoff_to_b = handoff(b)
        a.tools[a_handoff_to_b.to_tool(a).name] = a_handoff_to_b.to_tool(a)
        a._handoffs.append(a_handoff_to_b)

        fake, _ = _scripted_provider({
            "A": ToolCall(tool="transfer_to_b", args={}),
            "B": ToolCall(tool="transfer_to_a", args={}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(CircularHandoffError):
            a.run("start")

    def test_max_depth_exceeded(self, monkeypatch):
        monkeypatch.setattr(handoffs_mod, "MAX_HANDOFF_DEPTH", 2)
        d = make_agent("D")
        c = make_agent("C", handoffs=[d])
        b = make_agent("B", handoffs=[c])
        a = make_agent("A", handoffs=[b])

        fake, _ = _scripted_provider({
            "A": ToolCall(tool="transfer_to_b", args={}),
            "B": ToolCall(tool="transfer_to_c", args={}),
            "C": ToolCall(tool="transfer_to_d", args={}),
            "D": "should never be reached",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(MaxHandoffDepthExceededError):
            a.run("start")


class TestInvalidTarget:
    def test_invalid_target_raises_at_construction(self):
        with pytest.raises(InvalidHandoffTargetError):
            handoff("not an agent")

    def test_none_target_raises_at_construction(self):
        with pytest.raises(InvalidHandoffTargetError):
            handoff(None)

    def test_invalid_target_in_handoffs_list_raises_at_agent_construction(self):
        with pytest.raises(InvalidHandoffTargetError):
            make_agent("Bad", handoffs=["not an agent"])


class TestMemoryAndContext:
    def test_target_memory_contains_prior_conversation(self, monkeypatch):
        billing = make_agent("Billing")
        support = make_agent("Support", handoffs=[billing])
        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={"message": "context note"}),
            "Billing": "ok",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        support.run("original user question")
        prompt_text = billing.memory.to_prompt()
        assert "original user question" in prompt_text

    def test_shared_memory_does_not_duplicate_messages(self, monkeypatch):
        from abzagent.core.memory import Memory
        shared = Memory()
        billing = make_agent("Billing", memory=shared)
        support = make_agent("Support", memory=shared, handoffs=[billing])

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": "ok",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        support.run("hello")
        # count occurrences of the original user message in the rendered prompt —
        # must appear exactly once, not duplicated by the replay step
        prompt_text = shared.to_prompt()
        assert prompt_text.count("hello") == 1

    def test_context_forwarded_through_handoff(self, monkeypatch):
        received = []

        def on_handoff(ctx, *args):
            received.append(ctx.context)

        billing = make_agent("Billing")
        support = make_agent(
            "Support",
            handoffs=[handoff(billing, on_handoff=on_handoff)],
        )
        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": "ok",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        sentinel = {"user_id": 7}
        support.run("hi", context=sentinel)
        assert received == [sentinel]


class TestToolCompatibility:
    def test_target_agents_own_tools_still_work_after_handoff(self, monkeypatch):
        def lookup_price(item: str) -> str:
            """Look up a price. Args: item: the item name"""
            return f"${item} costs 10"

        billing = make_agent("Billing", tools=[lookup_price], max_iterations=3)
        support = make_agent("Support", handoffs=[billing])

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": [
                ToolCall(tool="lookup_price", args={"item": "widget"}),
                "The widget costs $10.",
            ],
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = support.run("how much is a widget")
        assert result.content == "The widget costs $10."
        assert result.last_agent is billing

    def test_unknown_tool_name_still_falls_through_to_execute_tool(self, monkeypatch):
        """Regression guard: a hallucinated, non-handoff tool name must still hit
        the existing '_execute_tool' Unknown-tool path, not the handoff branch."""
        def noop_tool(x: str) -> str:
            """A no-op tool. Args: x: anything"""
            return x

        agent = make_agent("Solo", tools=[noop_tool])
        fake, _ = _scripted_provider({
            "Solo": ToolCall(tool="totally_made_up_tool", args={}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("hi")
        assert "Unknown tool" in result.content
        assert result.last_agent is agent


class TestStructuredOutputCompatibility:
    def test_parsed_populated_after_handoff(self, monkeypatch):
        class Quote(BaseModel):
            item: str
            price: int

        billing = make_agent("Billing", output_type=Quote)
        support = make_agent("Support", handoffs=[billing])

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": '{"item": "widget", "price": 10}',
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = support.run("quote a widget")
        assert isinstance(result.parsed, Quote)
        assert result.parsed.item == "widget"
        assert result.parsed.price == 10
        assert result.last_agent is billing


class TestBackwardCompatibility:
    def test_agent_result_without_last_agent_still_works(self):
        r = AgentResult(content="hi", steps=["hi"])
        assert r.content == "hi"
        assert r.steps == ["hi"]
        assert r.parsed is None
        assert r.last_agent is None

    def test_run_signature_unchanged(self, monkeypatch):
        agent = make_agent("Solo")
        fake, _ = _scripted_provider({"Solo": "answer"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)
        # positional user_message, keyword-only context — exactly as before
        result = agent.run("question", context={"a": 1})
        assert result.content == "answer"

    def test_as_tool_unaffected_by_run_split(self, monkeypatch):
        sub = make_agent("Summarizer")
        fake, _ = _scripted_provider({"Summarizer": "a summary"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        tool = sub.as_tool(tool_name="summarize", tool_description="Summarize text.")
        assert tool.run(message="long text") == "a summary"


class TestOnHandoffAndInputType:
    def test_input_type_validated_and_passed_to_on_handoff(self, monkeypatch):
        class EscalationData(BaseModel):
            reason: str

        received = {}

        def on_handoff(ctx, data):
            received["reason"] = data.reason
            received["target_agent"] = ctx.target_agent

        escalation = make_agent("Escalation")
        support = make_agent(
            "Support",
            handoffs=[handoff(escalation, input_type=EscalationData, on_handoff=on_handoff)],
        )

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_escalation", args={"reason": "angry customer"}),
            "Escalation": "handled",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = support.run("I'm furious")
        assert received["reason"] == "angry customer"
        assert received["target_agent"] is escalation
        assert result.last_agent is escalation

    def test_missing_required_input_type_field_degrades_gracefully(self, monkeypatch):
        class EscalationData(BaseModel):
            reason: str  # required, model will omit it below

        escalation = make_agent("Escalation")
        support = make_agent(
            "Support",
            handoffs=[handoff(escalation, input_type=EscalationData)],
        )

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_escalation", args={}),  # missing "reason"
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = support.run("help")
        assert "information to proceed" in result.content
        assert result.last_agent is support  # handoff did NOT happen


class TestInputFilter:
    def test_remove_all_tools_filters_target_memory(self, monkeypatch):
        def noop_tool(x: str) -> str:
            """A no-op tool. Args: x: anything"""
            return x

        billing = make_agent("Billing")
        support = make_agent(
            "Support",
            handoffs=[handoff(billing, input_filter=remove_all_tools)],
            tools=[noop_tool],
            max_iterations=3,
        )

        fake, _ = _scripted_provider({
            "Support": [
                ToolCall(tool="noop_tool", args={"x": "noise"}),
                ToolCall(tool="transfer_to_billing", args={}),
            ],
            "Billing": "ok",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        support.run("hi")
        prompt_text = billing.memory.to_prompt()
        assert "[TOOL]" not in prompt_text

    def test_keep_last_n_turns_limits_target_memory(self, monkeypatch):
        billing = make_agent("Billing")
        support = make_agent("Support", handoffs=[handoff(billing, input_filter=keep_last_n_turns(1))])

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": "ok",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        support.memory.remember("user", "old message 1")
        support.memory.remember("tool", "old tool result")
        support.run("latest message")

        prompt_text = billing.memory.to_prompt()
        assert "old message 1" not in prompt_text


class TestToolDescriptionAutoDerivation:
    def test_bare_agent_gets_rich_description_from_instructions(self):
        """Exact example from the request: zero extra config, still rich."""
        writer = Agent(name="Writer", instructions="Write blogs and articles.", model="gemini-2.0-flash")
        planner = Agent(name="Planner", instructions="Route tasks.", model="gemini-2.0-flash", handoffs=[writer])

        tool = planner.tools["transfer_to_writer"]
        assert "Write blogs and articles." in tool.description
        assert tool.description != "Transfer the conversation to the 'Writer' agent."

    def test_tool_description_override_still_wins(self):
        writer = make_agent("Writer")
        planner = make_agent(
            "Planner", handoffs=[handoff(writer, tool_description_override="Custom routing text.")]
        )
        tool = planner.tools["transfer_to_writer"]
        assert tool.description == "Custom routing text."

    def test_dynamic_instructions_fall_back_to_generic_description(self):
        def dynamic_instructions(ctx, agent):
            return "dynamic!"

        dyn_target = Agent(name="Dyn", instructions=dynamic_instructions, model="gemini-2.0-flash")
        host = make_agent("Host", handoffs=[dyn_target])
        tool = host.tools["transfer_to_dyn"]
        assert tool.description == "Transfer the conversation to the 'Dyn' agent."

    def test_long_instructions_are_truncated_not_verbatim(self):
        long_target = Agent(name="Long", instructions="x" * 500, model="gemini-2.0-flash")
        host = make_agent("Host", handoffs=[long_target])
        tool = host.tools["transfer_to_long"]
        assert "x" * 500 not in tool.description
        assert "..." in tool.description
        assert len(tool.description) < 400


class TestInteractiveHandoffVisualization:
    def test_no_output_during_plain_non_interactive_run(self, monkeypatch, capsys):
        """Core regression guard: the library must never print unsolicited
        output during a plain programmatic .run() call, even when a handoff
        actually occurs."""
        billing = make_agent("Billing")
        support = make_agent("Support", handoffs=[billing])
        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": "ok",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = support.run("hi")
        assert result.last_agent is billing
        assert capsys.readouterr().out == ""

    def test_diagram_shown_during_interactive_handoff(self, monkeypatch, capsys):
        billing = make_agent("Billing")
        support = make_agent("Support", handoffs=[billing])
        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={}),
            "Billing": "ok",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        it = iter(["hi", "exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))

        support.run(interactive=True)
        out = capsys.readouterr().out
        assert "🔄 Handoff" in out
        assert "Support" in out
        assert "Billing" in out
        assert "│" in out
        assert "▼" in out

    def test_multi_hop_chain_prints_sequential_diagrams_not_one_cumulative(self, monkeypatch, capsys):
        c = make_agent("C")
        b = make_agent("B", handoffs=[c])
        a = make_agent("A", handoffs=[b])

        fake, _ = _scripted_provider({
            "A": ToolCall(tool="transfer_to_b", args={}),
            "B": ToolCall(tool="transfer_to_c", args={}),
            "C": "final answer",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        it = iter(["start", "exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))

        a.run(interactive=True)
        out = capsys.readouterr().out

        assert out.count("🔄 Handoff") == 2
        second_block_start = out.index("🔄 Handoff", out.index("🔄 Handoff") + 1)
        first_block, second_block = out[:second_block_start], out[second_block_start:]

        assert "A\n" in first_block and "B\n" in first_block
        assert "B\n" in second_block and "C\n" in second_block
