"""
Provider-level unit tests for native tool-calling translation, mocking the raw
SDK client calls directly (not .generate() itself) — verifies ToolSchema ->
native request shaping and native response -> ToolCall mapping in isolation.
"""
import json
import os

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent.config import SDKConfig
from abzagent.core.tools import ToolSchema
from abzagent.providers.gemini import GeminiProvider
from abzagent.providers.groq import GroqProvider


def _make_gemini_provider():
    cfg = SDKConfig(model="gemini-2.5-flash", api_key="fake", provider="gemini")
    return GeminiProvider(cfg)


def _make_groq_provider():
    cfg = SDKConfig(model="llama-3.3-70b-versatile", api_key="fake", provider="groq")
    return GroqProvider(cfg)


class _FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakeGeminiResponse:
    def __init__(self, function_calls=None, text=""):
        self.function_calls = function_calls
        self.text = text


class TestGeminiNativeToolCalling:
    def test_tool_schema_translated_into_function_declaration(self, monkeypatch):
        provider = _make_gemini_provider()
        captured = {}

        def fake_generate_content(*, model, contents, config):
            captured["config"] = config
            return _FakeGeminiResponse(text="no call")

        monkeypatch.setattr(provider.client.models, "generate_content", fake_generate_content)

        schema = ToolSchema(
            name="lookup_price",
            description="Look up a price.",
            parameters={"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
        )
        provider.generate("prompt", tools=[schema])

        config = captured["config"]
        assert config.tools is not None
        decl = config.tools[0].function_declarations[0]
        assert decl.name == "lookup_price"
        assert decl.parameters_json_schema == schema.parameters
        assert config.automatic_function_calling.disable is True

    def test_function_call_response_mapped_to_tool_calls(self, monkeypatch):
        provider = _make_gemini_provider()

        def fake_generate_content(*, model, contents, config):
            return _FakeGeminiResponse(
                function_calls=[_FakeFunctionCall(name="lookup_price", args={"item": "widget"})]
            )

        monkeypatch.setattr(provider.client.models, "generate_content", fake_generate_content)

        result = provider.generate("prompt", tools=[ToolSchema(name="lookup_price", description="", parameters={})])
        assert result.text is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool == "lookup_price"
        assert result.tool_calls[0].args == {"item": "widget"}

    def test_multiple_function_calls_all_mapped(self, monkeypatch):
        provider = _make_gemini_provider()

        def fake_generate_content(*, model, contents, config):
            return _FakeGeminiResponse(
                function_calls=[
                    _FakeFunctionCall(name="a", args={}),
                    _FakeFunctionCall(name="b", args={}),
                ]
            )

        monkeypatch.setattr(provider.client.models, "generate_content", fake_generate_content)
        result = provider.generate("prompt", tools=[ToolSchema(name="a", description="", parameters={})])
        assert [tc.tool for tc in result.tool_calls] == ["a", "b"]

    def test_text_only_response_unchanged(self, monkeypatch):
        provider = _make_gemini_provider()

        def fake_generate_content(*, model, contents, config):
            return _FakeGeminiResponse(text="a plain answer")

        monkeypatch.setattr(provider.client.models, "generate_content", fake_generate_content)
        result = provider.generate("prompt")
        assert result.text == "a plain answer"
        assert result.tool_calls == []

    def test_no_tools_does_not_set_tools_config(self, monkeypatch):
        provider = _make_gemini_provider()
        captured = {}

        def fake_generate_content(*, model, contents, config):
            captured["config"] = config
            return _FakeGeminiResponse(text="ok")

        monkeypatch.setattr(provider.client.models, "generate_content", fake_generate_content)
        provider.generate("prompt")
        assert captured["config"] is None


class _FakeGroqFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeGroqToolCall:
    def __init__(self, name, arguments):
        self.function = _FakeGroqFunction(name, arguments)


class _FakeGroqMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeGroqChoice:
    def __init__(self, message):
        self.message = message


class _FakeGroqCompletion:
    def __init__(self, message):
        self.choices = [_FakeGroqChoice(message)]


class TestGroqNativeToolCalling:
    def test_tool_schema_translated_into_openai_tool_shape(self, monkeypatch):
        provider = _make_groq_provider()
        captured = {}

        def fake_create(**kwargs):
            captured["kwargs"] = kwargs
            return _FakeGroqCompletion(_FakeGroqMessage(content="no call"))

        monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

        schema = ToolSchema(
            name="lookup_price",
            description="Look up a price.",
            parameters={"type": "object", "properties": {"item": {"type": "string"}}},
        )
        provider.generate("prompt", tools=[schema])

        tools = captured["kwargs"]["tools"]
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "lookup_price"
        assert tools[0]["function"]["parameters"] == schema.parameters

    def test_tool_calls_response_parses_json_arguments(self, monkeypatch):
        provider = _make_groq_provider()

        def fake_create(**kwargs):
            return _FakeGroqCompletion(_FakeGroqMessage(tool_calls=[
                _FakeGroqToolCall("lookup_price", json.dumps({"item": "widget"}))
            ]))

        monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)
        result = provider.generate("prompt", tools=[ToolSchema(name="lookup_price", description="", parameters={})])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool == "lookup_price"
        assert result.tool_calls[0].args == {"item": "widget"}

    def test_malformed_arguments_json_falls_back_to_empty_dict(self, monkeypatch):
        provider = _make_groq_provider()

        def fake_create(**kwargs):
            return _FakeGroqCompletion(_FakeGroqMessage(tool_calls=[
                _FakeGroqToolCall("lookup_price", "{not valid json")
            ]))

        monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)
        result = provider.generate("prompt", tools=[ToolSchema(name="lookup_price", description="", parameters={})])

        assert result.tool_calls[0].tool == "lookup_price"
        assert result.tool_calls[0].args == {}

    def test_text_only_response_unchanged(self, monkeypatch):
        provider = _make_groq_provider()

        def fake_create(**kwargs):
            return _FakeGroqCompletion(_FakeGroqMessage(content="a plain answer"))

        monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)
        result = provider.generate("prompt")
        assert result.text == "a plain answer"
        assert result.tool_calls == []

    def test_exception_still_returns_error_text_not_raise(self, monkeypatch):
        provider = _make_groq_provider()

        def fake_create(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)
        result = provider.generate("prompt")
        assert result.tool_calls == []
        assert "[Groq Error] boom" in result.text
