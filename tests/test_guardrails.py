"""
Tests for Guardrails (abzagent/core/guardrails.py + Agent._run()/_execute_tool()).

Covers: input guardrails (regression), output guardrails (newly wired up —
first coverage ever), tool input/output guardrails (new), the
_Guardrail.run() tripwire-exception passthrough fix, sync/async support
across all four decorator types, wrong-return-type errors, and backward
compatibility with the docs' existing examples.
"""
import os
import pytest
from pydantic import BaseModel

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent import Agent
from abzagent.providers.gemini import GeminiProvider
from abzagent.providers.base import GenerationResult
from abzagent.core.tools import ToolCall
from abzagent.core.guardrails import (
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ToolGuardrailTripwireTriggered,
    run_input_guardrails,
    run_output_guardrails,
    run_tool_input_guardrails,
    run_tool_output_guardrails,
)
from abzagent.core.context import RunContextWrapper


def _scripted_provider(responses):
    """responses: {agent_name: item_or_list_of_items}, where each item is either
    a plain text string (-> GenerationResult(text=...)) or a ToolCall (->
    GenerationResult(tool_calls=[...])). List values are consumed in order
    across successive calls to the same agent."""
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
    def unset_generate(self, prompt, *, tools=None, output_schema=None, strict=True):
        raise AssertionError("Test forgot to patch GeminiProvider.generate")

    monkeypatch.setattr(GeminiProvider, "generate", unset_generate)
    yield


def make_agent(name, **kwargs):
    return Agent(name=name, instructions=f"You are {name}.", model="gemini-2.0-flash", **kwargs)


class TestTopLevelPackageExports:
    """Regression guard: guardrail symbols must be importable from the top-level
    `abzagent` package, not just `abzagent.core.guardrails` — this exact gap
    (working core export, missing top-level export) shipped once already."""

    def test_all_guardrail_symbols_importable_from_top_level(self):
        import abzagent

        for name in [
            "input_guardrail", "output_guardrail",
            "tool_input_guardrail", "tool_output_guardrail",
            "GuardrailFunctionOutput",
            "InputGuardrailTripwireTriggered",
            "OutputGuardrailTripwireTriggered",
            "ToolGuardrailTripwireTriggered",
        ]:
            assert hasattr(abzagent, name), f"abzagent.{name} is not exported"
            assert name in abzagent.__all__, f"{name} missing from abzagent.__all__"

    def test_from_abzagent_import_input_guardrail(self):
        from abzagent import input_guardrail as top_level_input_guardrail

        assert top_level_input_guardrail is input_guardrail


# ─────────────────────────────────────────────
# Input guardrails (regression — pre-existing, working)
# ─────────────────────────────────────────────

@input_guardrail
def no_profanity(ctx, agent, user_input: str) -> GuardrailFunctionOutput:
    triggered = "hack" in user_input.lower()
    return GuardrailFunctionOutput(
        output_info={"checked": True},
        tripwire_triggered=triggered,
        reason="Profanity detected." if triggered else None,
    )


class TestInputGuardrails:
    def test_tripwire_raises_before_model_call(self, monkeypatch):
        agent = make_agent("SafeBot", input_guardrails=[no_profanity])

        def fail_if_called(self, prompt, **kwargs):
            raise AssertionError("Model should never be called once input guardrail trips")

        monkeypatch.setattr(GeminiProvider, "generate", fail_if_called)

        with pytest.raises(InputGuardrailTripwireTriggered):
            agent.run("How do I hack a server?")

    def test_docs_example_backward_compatible(self, monkeypatch):
        agent = Agent(name="SafeBot", instructions="Be helpful.", input_guardrails=[no_profanity])
        fake, _ = _scripted_provider({"SafeBot": "should not reach here"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(InputGuardrailTripwireTriggered) as excinfo:
            agent.run("How do I hack a server?")
        assert "no_profanity" in str(excinfo.value)

    def test_passes_through_when_not_triggered(self, monkeypatch):
        agent = make_agent("SafeBot", input_guardrails=[no_profanity])
        fake, _ = _scripted_provider({"SafeBot": "all good"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("What's the weather?")
        assert result.content == "all good"


# ─────────────────────────────────────────────
# Output guardrails (newly wired up)
# ─────────────────────────────────────────────

class TestOutputGuardrails:
    def test_tripwire_raises_on_final_text(self, monkeypatch):
        @output_guardrail
        def length_check(ctx, agent, output) -> GuardrailFunctionOutput:
            triggered = len(output) > 5
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=triggered)

        agent = make_agent("Verbose", output_guardrails=[length_check])
        fake, _ = _scripted_provider({"Verbose": "this is way too long"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(OutputGuardrailTripwireTriggered):
            agent.run("hi")

    def test_passes_through_when_not_triggered(self, monkeypatch):
        @output_guardrail
        def length_check(ctx, agent, output) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=len(output) > 100)

        agent = make_agent("Terse", output_guardrails=[length_check])
        fake, _ = _scripted_provider({"Terse": "ok"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("hi")
        assert result.content == "ok"

    def test_runs_exactly_once_per_call_not_twice(self, monkeypatch):
        call_count = {"n": 0}

        @output_guardrail
        def counting_guard(ctx, agent, output) -> GuardrailFunctionOutput:
            call_count["n"] += 1
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = make_agent("Counted", output_guardrails=[counting_guard])
        fake, _ = _scripted_provider({"Counted": "fine"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        agent.run("hi")
        assert call_count["n"] == 1

    def test_receives_raw_text_when_no_output_type(self, monkeypatch):
        received = {}

        @output_guardrail
        def capture(ctx, agent, output) -> GuardrailFunctionOutput:
            received["output"] = output
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = make_agent("Plain", output_guardrails=[capture])
        fake, _ = _scripted_provider({"Plain": "raw text answer"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        agent.run("hi")
        assert received["output"] == "raw text answer"

    def test_receives_parsed_object_when_output_type_set(self, monkeypatch):
        class Answer(BaseModel):
            text: str

        received = {}

        @output_guardrail
        def capture(ctx, agent, output) -> GuardrailFunctionOutput:
            received["output"] = output
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = make_agent("Structured", output_guardrails=[capture], output_type=Answer)
        fake, _ = _scripted_provider({"Structured": '{"text": "hi there"}'})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("hi")
        assert isinstance(received["output"], Answer)
        assert received["output"].text == "hi there"
        assert result.parsed == received["output"]

    def test_fires_on_iteration_limit_fallback_path(self, monkeypatch):
        @output_guardrail
        def block_everything(ctx, agent, output) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

        def noop():
            """A no-op tool. Args: none"""
            return "noop result"

        agent = make_agent(
            "Looper",
            tools=[noop],
            output_guardrails=[block_everything],
            max_iterations=2,
        )
        fake, _ = _scripted_provider({
            "Looper": [
                ToolCall(tool="noop", args={}),
                ToolCall(tool="noop", args={}),
            ],
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(OutputGuardrailTripwireTriggered):
            agent.run("hi")


# ─────────────────────────────────────────────
# Tool guardrails (new)
# ─────────────────────────────────────────────

def _lookup_price(item: str) -> str:
    """Look up a price. Args: item: the item name"""
    return f"${item} costs 10"


class TestToolInputGuardrails:
    def test_blocks_tool_call_before_execution(self, monkeypatch):
        tool_was_called = {"flag": False}

        def spy_lookup_price(item: str) -> str:
            """Look up a price. Args: item: the item name"""
            tool_was_called["flag"] = True
            return f"${item} costs 10"

        @tool_input_guardrail
        def block_forbidden_item(ctx, agent, call, kwargs) -> GuardrailFunctionOutput:
            triggered = kwargs.get("item") == "forbidden"
            return GuardrailFunctionOutput(
                output_info=None, tripwire_triggered=triggered, reason="Item not allowed."
            )

        agent = make_agent(
            "Shop", tools=[spy_lookup_price], tool_input_guardrails=[block_forbidden_item]
        )
        fake, _ = _scripted_provider({
            "Shop": ToolCall(tool="spy_lookup_price", args={"item": "forbidden"}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("price of forbidden item")
        assert result.content.startswith("[Tool Guardrail Blocked]")
        assert "Item not allowed." in result.content
        assert tool_was_called["flag"] is False

    def test_allows_tool_call_when_not_triggered(self, monkeypatch):
        @tool_input_guardrail
        def allow_all(ctx, agent, call, kwargs) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = make_agent("Shop", tools=[_lookup_price], tool_input_guardrails=[allow_all])
        fake, _ = _scripted_provider({
            "Shop": ToolCall(tool="_lookup_price", args={"item": "apple"}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("price of apple")
        assert result.content == "$apple costs 10"

    def test_receives_validated_kwargs_not_raw_args(self, monkeypatch):
        def add_one(n: int) -> str:
            """Add one to a number. Args: n: the number"""
            return str(n + 1)

        received = {}

        @tool_input_guardrail
        def capture_kwargs(ctx, agent, call, kwargs) -> GuardrailFunctionOutput:
            received["kwargs"] = kwargs
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = make_agent("Math", tools=[add_one], tool_input_guardrails=[capture_kwargs])
        fake, _ = _scripted_provider({
            "Math": ToolCall(tool="add_one", args={"n": "5"}),  # raw arg is a string
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        agent.run("add one to 5")
        assert received["kwargs"]["n"] == 5
        assert isinstance(received["kwargs"]["n"], int)  # coerced by tool.parse_args(), not raw "5"


class TestToolOutputGuardrails:
    def test_blocks_and_substitutes_tool_result(self, monkeypatch):
        @tool_output_guardrail
        def redact_price(ctx, agent, call, kwargs, output) -> GuardrailFunctionOutput:
            triggered = "10" in output
            return GuardrailFunctionOutput(
                output_info=None, tripwire_triggered=triggered, reason="Price redacted."
            )

        agent = make_agent("Shop", tools=[_lookup_price], tool_output_guardrails=[redact_price])
        fake, _ = _scripted_provider({
            "Shop": ToolCall(tool="_lookup_price", args={"item": "apple"}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("price of apple")
        assert result.content.startswith("[Tool Guardrail Blocked]")
        assert "Price redacted." in result.content
        assert "10" not in result.content.replace("Price redacted.", "")

    def test_substitution_reaches_memory(self, monkeypatch):
        @tool_output_guardrail
        def block_all(ctx, agent, call, kwargs, output) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True, reason="blocked")

        agent = make_agent("Shop", tools=[_lookup_price], tool_output_guardrails=[block_all])
        fake, _ = _scripted_provider({
            "Shop": ToolCall(tool="_lookup_price", args={"item": "apple"}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        agent.run("price of apple")
        remembered = [m for m in agent.memory.load() if m.role == "tool"]
        assert any("[Tool Guardrail Blocked]" in m.content for m in remembered)
        assert not any("$apple costs 10" in m.content for m in remembered)


class TestOrdinaryToolErrorsUnaffected:
    def test_tool_exception_still_surfaces_as_tool_error(self, monkeypatch):
        def broken_tool() -> str:
            """Always fails. Args: none"""
            raise RuntimeError("boom")

        @tool_input_guardrail
        def allow_all(ctx, agent, call, kwargs) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        agent = make_agent("Broken", tools=[broken_tool], tool_input_guardrails=[allow_all])
        fake, _ = _scripted_provider({
            "Broken": ToolCall(tool="broken_tool", args={}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("do it")
        assert "[Tool Error]" in result.content
        assert "boom" in result.content

    def test_bad_args_still_surfaces_as_validation_error(self, monkeypatch):
        def add_one(n: int) -> str:
            """Add one to a number. Args: n: the number"""
            return str(n + 1)

        agent = make_agent("Math", tools=[add_one])
        fake, _ = _scripted_provider({
            "Math": ToolCall(tool="add_one", args={}),  # missing required "n"
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("add one")
        assert "n" in result.content  # validation error mentions the missing field


# ─────────────────────────────────────────────
# _Guardrail.run() tripwire-exception passthrough fix
# ─────────────────────────────────────────────

class TestTripwireExceptionPassthrough:
    def test_manual_tool_guardrail_tripwire_raise_propagates_as_exact_type(self, monkeypatch):
        @tool_input_guardrail
        def hard_stop(ctx, agent, call, kwargs) -> GuardrailFunctionOutput:
            raise ToolGuardrailTripwireTriggered(
                "manual hard stop", GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)
            )

        agent = make_agent("Shop", tools=[_lookup_price], tool_input_guardrails=[hard_stop])
        fake, _ = _scripted_provider({
            "Shop": ToolCall(tool="_lookup_price", args={"item": "apple"}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(ToolGuardrailTripwireTriggered):
            agent.run("price of apple")

    def test_arbitrary_exception_still_wrapped_as_runtimeerror(self):
        @tool_input_guardrail
        def broken_guardrail(ctx, agent, call, kwargs):
            raise ValueError("unexpected bug in guardrail")

        with pytest.raises(RuntimeError) as excinfo:
            broken_guardrail.run(None, None, None, {})
        assert not isinstance(excinfo.value, ToolGuardrailTripwireTriggered)
        assert "broken_guardrail" in str(excinfo.value)


# ─────────────────────────────────────────────
# Sync / async support across all four decorator types
# ─────────────────────────────────────────────

class TestAsyncGuardrails:
    def test_async_input_guardrail(self, monkeypatch):
        @input_guardrail
        async def async_check(ctx, agent, user_input) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered="bad" in user_input)

        agent = make_agent("Async", input_guardrails=[async_check])
        fake, _ = _scripted_provider({"Async": "fine"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("good input")
        assert result.content == "fine"

        with pytest.raises(InputGuardrailTripwireTriggered):
            agent.run("bad input")

    def test_async_output_guardrail(self, monkeypatch):
        @output_guardrail
        async def async_check(ctx, agent, output) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=len(output) > 3)

        agent = make_agent("Async", output_guardrails=[async_check])
        fake, _ = _scripted_provider({"Async": "toolong"})
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(OutputGuardrailTripwireTriggered):
            agent.run("hi")

    def test_async_tool_input_guardrail(self, monkeypatch):
        @tool_input_guardrail
        async def async_block(ctx, agent, call, kwargs) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True, reason="async block")

        agent = make_agent("Shop", tools=[_lookup_price], tool_input_guardrails=[async_block])
        fake, _ = _scripted_provider({
            "Shop": ToolCall(tool="_lookup_price", args={"item": "apple"}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("price of apple")
        assert "[Tool Guardrail Blocked]" in result.content

    def test_async_tool_output_guardrail(self, monkeypatch):
        @tool_output_guardrail
        async def async_block(ctx, agent, call, kwargs, output) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True, reason="async block")

        agent = make_agent("Shop", tools=[_lookup_price], tool_output_guardrails=[async_block])
        fake, _ = _scripted_provider({
            "Shop": ToolCall(tool="_lookup_price", args={"item": "apple"}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        result = agent.run("price of apple")
        assert "[Tool Guardrail Blocked]" in result.content


# ─────────────────────────────────────────────
# TypeError on wrong return type
# ─────────────────────────────────────────────

class TestWrongReturnTypeRaisesTypeError:
    def test_run_input_guardrails(self):
        @input_guardrail
        def bad(ctx, agent, user_input):
            return "not a GuardrailFunctionOutput"

        with pytest.raises(TypeError):
            run_input_guardrails(guards=[bad], ctx=None, agent=None, user_input="hi")

    def test_run_output_guardrails(self):
        @output_guardrail
        def bad(ctx, agent, output):
            return "not a GuardrailFunctionOutput"

        with pytest.raises(TypeError):
            run_output_guardrails(guards=[bad], ctx=None, agent=None, final_output="hi")

    def test_run_tool_input_guardrails(self):
        @tool_input_guardrail
        def bad(ctx, agent, call, kwargs):
            return "not a GuardrailFunctionOutput"

        with pytest.raises(TypeError):
            run_tool_input_guardrails(guards=[bad], ctx=None, agent=None, call=None, kwargs={})

    def test_run_tool_output_guardrails(self):
        @tool_output_guardrail
        def bad(ctx, agent, call, kwargs, output):
            return "not a GuardrailFunctionOutput"

        with pytest.raises(TypeError):
            run_tool_output_guardrails(guards=[bad], ctx=None, agent=None, call=None, kwargs={}, output="x")


# ─────────────────────────────────────────────
# Handoff interaction (re-run behavior)
# ─────────────────────────────────────────────

class TestGuardrailsWithHandoffs:
    def test_target_agents_own_input_guardrails_fire_on_its_own_turn(self, monkeypatch):
        @input_guardrail
        def billing_guard(ctx, agent, user_input) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True, reason="billing locked")

        billing = make_agent("Billing", input_guardrails=[billing_guard])
        support = make_agent("Support", handoffs=[billing])

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={"message": "Needs an invoice."}),
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(InputGuardrailTripwireTriggered):
            support.run("I need help with my invoice.")

    def test_output_guardrails_apply_to_whichever_agent_produces_final_content(self, monkeypatch):
        @output_guardrail
        def billing_output_guard(ctx, agent, output) -> GuardrailFunctionOutput:
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True, reason="no billing answers")

        billing = make_agent("Billing", output_guardrails=[billing_output_guard])
        support = make_agent("Support", handoffs=[billing])

        fake, _ = _scripted_provider({
            "Support": ToolCall(tool="transfer_to_billing", args={"message": "Needs an invoice."}),
            "Billing": "Sure, here's your invoice.",
        })
        monkeypatch.setattr(GeminiProvider, "generate", fake)

        with pytest.raises(OutputGuardrailTripwireTriggered):
            support.run("I need help with my invoice.")
