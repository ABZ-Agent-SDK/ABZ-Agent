"""
Tests for the native-tool-calling / fallback split (Agent._build_prompt()'s
gating on provider.supports_native_tools).

Provider-level translation logic (ToolSchema -> native request, native
response -> ToolCall) is covered separately in tests/test_providers.py.
"""
import os
import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent import Agent
from abzagent.providers.gemini import GeminiProvider


def _make_tool_agent(model="gemini-2.0-flash", **kwargs):
    def lookup_price(item: str) -> str:
        """Look up a price. Args: item: the item name"""
        return f"${item} costs 10"

    return Agent(
        name="Shop",
        instructions="Help with prices.",
        model=model,
        tools=[lookup_price],
        **kwargs,
    )


class TestNativeModePromptOmitsManifest:
    def test_no_tool_manifest_text_in_native_mode(self):
        agent = _make_tool_agent()
        assert agent.provider.supports_native_tools is True

        prompt = agent._build_prompt("hi", effective_instructions=agent.instructions or "Be helpful.")
        assert "Available TOOLS:" not in prompt
        assert '{"tool":"<name>"' not in prompt  # JSON-blob teaching also omitted

    def test_handoff_hint_is_the_native_variant(self):
        specialist = Agent(name="Specialist", instructions="Handle specialist work.", model="gemini-2.0-flash")
        agent = Agent(
            name="Router",
            instructions="Route work.",
            model="gemini-2.0-flash",
            handoffs=[specialist],
        )
        prompt = agent._build_prompt("hi", effective_instructions=agent.instructions or "Route work.")
        assert "You are a ROUTER first" in prompt  # native hint
        assert 'Return ONLY a JSON object for the tool call' not in prompt  # fallback hint's JSON instructions


class TestFallbackModePromptIncludesManifest:
    def test_tool_manifest_and_json_blob_teaching_present_when_not_native(self, monkeypatch):
        agent = _make_tool_agent()
        # Simulate a hypothetical provider without native tool-calling support.
        monkeypatch.setattr(type(agent.provider), "supports_native_tools", False)

        prompt = agent._build_prompt("hi", effective_instructions=agent.instructions or "Be helpful.")
        assert "Available TOOLS:" in prompt
        assert "lookup_price" in prompt
        assert '{"tool":"<name>","args":{...}}' in prompt

    def test_handoff_hint_is_the_fallback_variant_when_not_native(self, monkeypatch):
        specialist = Agent(name="Specialist", instructions="Handle specialist work.", model="gemini-2.0-flash")
        agent = Agent(
            name="Router",
            instructions="Route work.",
            model="gemini-2.0-flash",
            handoffs=[specialist],
        )
        monkeypatch.setattr(type(agent.provider), "supports_native_tools", False)

        prompt = agent._build_prompt("hi", effective_instructions=agent.instructions or "Route work.")
        assert "transfer_to_<agent_name_slug>" in prompt
        assert 'Return ONLY a JSON object for the tool call' in prompt


class TestHandoffHintPosition:
    def test_handoff_hint_comes_after_history_and_before_user_message(self):
        """The handoff hint is deliberately the last thing before [USER]: —
        models weight the end of the prompt more heavily, so this positioning
        matters for getting weaker models to actually transfer."""
        specialist = Agent(name="Specialist", instructions="Handle specialist work.", model="gemini-2.0-flash")
        agent = Agent(
            name="Router",
            instructions="Route work.",
            model="gemini-2.0-flash",
            handoffs=[specialist],
        )
        agent.memory.remember("user", "MARKER_HISTORY_LINE")

        prompt = agent._build_prompt("hi", effective_instructions="Route work.")
        history_pos = prompt.index("MARKER_HISTORY_LINE")
        hint_pos = prompt.index("You are a ROUTER first")
        final_user_message_pos = prompt.rindex("[USER]: hi")  # the actual trailing user message,
        # not the "[USER]: MARKER_HISTORY_LINE" line rendered as part of history

        assert history_pos < hint_pos < final_user_message_pos

    def test_no_handoffs_no_hint_at_all(self):
        agent = Agent(name="Solo", instructions="Be helpful.", model="gemini-2.0-flash")
        prompt = agent._build_prompt("hi", effective_instructions="Be helpful.")
        assert "ROUTER first" not in prompt


class TestGenerateAndDispatchToolsWiring:
    def test_native_mode_passes_tool_schemas_to_provider(self, monkeypatch):
        agent = _make_tool_agent()
        captured = {}

        def fake_generate(self, prompt, *, tools=None, output_schema=None, strict=True):
            captured["tools"] = tools
            from abzagent.providers.base import GenerationResult
            return GenerationResult(text="ok")

        monkeypatch.setattr(GeminiProvider, "generate", fake_generate)
        agent.run("hi")

        assert captured["tools"] is not None
        assert captured["tools"][0].name == "lookup_price"
        assert captured["tools"][0].parameters["type"] == "object"

    def test_no_tools_means_tools_arg_is_none(self, monkeypatch):
        agent = Agent(name="Solo", instructions="Be helpful.", model="gemini-2.0-flash")
        captured = {}

        def fake_generate(self, prompt, *, tools=None, output_schema=None, strict=True):
            captured["tools"] = tools
            from abzagent.providers.base import GenerationResult
            return GenerationResult(text="ok")

        monkeypatch.setattr(GeminiProvider, "generate", fake_generate)
        agent.run("hi")

        assert captured["tools"] is None
