"""
Tests for Agent.run(interactive=True) (abzagent/core/agent.py)

The SDK exposes a single execution method, run(). This covers both modes:
- run(user_message) — the normal single-request path (unchanged behavior)
- run(interactive=True) — a terminal REPL that is a thin wrapper around run()

Covers:
- Normal request still works exactly as before (backward compatibility)
- Interactive mode: banner + prompt loop mechanics
- exit / quit ends the session
- Ctrl+C (KeyboardInterrupt) during input() and during the run() call both exit cleanly
- EOF (e.g. piped stdin closing) exits cleanly instead of raising
- Blank input is skipped without calling run()
- An exception from run() is caught and printed, loop continues
- context is forwarded to run() unchanged
- Interactive mode reuses run() rather than duplicating agent logic
- Misuse (no args, or both user_message and interactive=True) raises ValueError
- chat() no longer exists
"""
import os
import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent import Agent
from abzagent.providers.gemini import GeminiProvider


@pytest.fixture
def agent(monkeypatch):
    def fake_generate(self, prompt, *, output_schema=None, strict=True):
        return "canned response"

    monkeypatch.setattr(GeminiProvider, "generate", fake_generate)
    return Agent(name="Assistant", instructions="Be helpful.", model="gemini-2.0-flash")


def _feed_input(monkeypatch, lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)


class TestNormalModeUnchanged:
    def test_run_with_positional_message_returns_agent_result(self, agent):
        result = agent.run("Hello")
        assert result.content == "canned response"

    def test_run_with_keyword_message_and_context(self, agent, monkeypatch):
        received = []
        original_run = agent._run

        def spy_run(user_message, *, context=None, _handoff_path=None):
            received.append((user_message, context))
            return original_run(user_message, context=context, _handoff_path=_handoff_path)

        agent._run = spy_run
        result = agent.run("Hello", context={"a": 1})
        assert result.content == "canned response"
        assert received == [("Hello", {"a": 1})]


class TestInteractiveBasicLoop:
    def test_banner_printed(self, agent, monkeypatch, capsys):
        _feed_input(monkeypatch, ["exit"])
        ret = agent.run(interactive=True)
        assert ret is None
        out = capsys.readouterr().out
        assert "Assistant started" in out

    def test_exit_ends_session(self, agent, monkeypatch, capsys):
        _feed_input(monkeypatch, ["exit"])
        agent.run(interactive=True)
        out = capsys.readouterr().out
        assert "Goodbye" in out

    def test_quit_ends_session(self, agent, monkeypatch, capsys):
        _feed_input(monkeypatch, ["quit"])
        agent.run(interactive=True)
        out = capsys.readouterr().out
        assert "Goodbye" in out

    def test_exit_case_insensitive(self, agent, monkeypatch, capsys):
        _feed_input(monkeypatch, ["EXIT"])
        agent.run(interactive=True)
        out = capsys.readouterr().out
        assert "Goodbye" in out

    def test_prints_agent_response(self, agent, monkeypatch, capsys):
        _feed_input(monkeypatch, ["hello", "exit"])
        agent.run(interactive=True)
        out = capsys.readouterr().out
        assert "Assistant: canned response" in out

    def test_multi_turn_conversation(self, agent, monkeypatch, capsys):
        _feed_input(monkeypatch, ["hi", "how are you", "exit"])
        agent.run(interactive=True)
        out = capsys.readouterr().out
        assert out.count("Assistant: canned response") == 2


class TestInteractiveBlankInput:
    def test_blank_input_skipped_without_calling_run(self, agent, monkeypatch, capsys):
        calls = []
        original_run = agent.run

        def spy_run(*a, **kw):
            calls.append((a, kw))
            return original_run(*a, **kw)

        agent.run = spy_run
        _feed_input(monkeypatch, ["", "   ", "exit"])
        agent.run(interactive=True)
        # only the final interactive=True call itself, no per-message calls
        assert calls == [((), {"interactive": True})]


class TestInteractiveInterruptsAndEOF:
    def test_keyboard_interrupt_during_input_exits_cleanly(self, agent, monkeypatch, capsys):
        def raising_input(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raising_input)
        agent.run(interactive=True)  # must not raise
        out = capsys.readouterr().out
        assert "Goodbye" in out

    def test_eof_exits_cleanly(self, agent, monkeypatch, capsys):
        _feed_input(monkeypatch, [])  # immediately raises EOFError
        agent.run(interactive=True)  # must not raise
        out = capsys.readouterr().out
        assert "Goodbye" in out

    def test_keyboard_interrupt_during_run_exits_cleanly(self, agent, monkeypatch, capsys):
        def interrupting_run(user_message, *, context=None):
            raise KeyboardInterrupt

        agent.run = interrupting_run
        _feed_input(monkeypatch, ["hello"])
        agent._run_interactive()  # must not raise
        out = capsys.readouterr().out
        assert "Goodbye" in out


class TestInteractiveErrorRecovery:
    def test_run_exception_is_caught_and_session_continues(self, agent, monkeypatch, capsys):
        calls = {"n": 0}
        original_run = agent.run

        def flaky_run(*a, **kw):
            if kw.get("interactive"):
                return original_run(*a, **kw)
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated transient failure")
            return original_run(*a, **kw)

        agent.run = flaky_run
        _feed_input(monkeypatch, ["first", "second", "exit"])
        agent.run(interactive=True)  # must not raise
        out = capsys.readouterr().out
        assert "[Error] simulated transient failure" in out
        assert "Assistant: canned response" in out
        assert calls["n"] == 2


class TestInteractiveContextForwarding:
    def test_context_forwarded_to_run(self, agent, monkeypatch):
        received = []
        original_run = agent.run

        def spy_run(*a, **kw):
            if not kw.get("interactive"):
                received.append(kw.get("context"))
            return original_run(*a, **kw)

        agent.run = spy_run
        _feed_input(monkeypatch, ["hello", "exit"])
        sentinel = {"user_id": 42}
        agent.run(interactive=True, context=sentinel)
        assert received == [sentinel]


class TestInteractiveReusesRun:
    def test_interactive_calls_run_not_duplicated_logic(self, agent, monkeypatch):
        """Interactive mode must delegate to run() rather than reimplementing agent logic."""
        called_with = []
        original_run = agent.run

        def spy_run(*a, **kw):
            if not kw.get("interactive") and a:
                called_with.append(a[0])
            return original_run(*a, **kw)

        agent.run = spy_run
        _feed_input(monkeypatch, ["what is 2+2", "exit"])
        agent.run(interactive=True)
        assert called_with == ["what is 2+2"]


class TestMisuseRaisesValueError:
    def test_no_args_raises(self, agent):
        with pytest.raises(ValueError):
            agent.run()

    def test_message_and_interactive_together_raises(self, agent):
        with pytest.raises(ValueError):
            agent.run("hi", interactive=True)


class TestChatRemoved:
    def test_chat_method_no_longer_exists(self, agent):
        assert not hasattr(agent, "chat")
