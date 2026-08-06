# abzagent/core/guardrails.py
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Sequence, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .agent import Agent  # type hint only; not executed at import time
    from .tools import ToolCall


class GuardrailFunctionOutput(BaseModel):
    """Result returned by a guardrail function."""
    output_info: Any
    tripwire_triggered: bool
    reason: Optional[str] = None


class InputGuardrailTripwireTriggered(RuntimeError):
    def __init__(self, message: str, output: GuardrailFunctionOutput):
        super().__init__(message)
        self.output = output


class OutputGuardrailTripwireTriggered(RuntimeError):
    def __init__(self, message: str, output: GuardrailFunctionOutput):
        super().__init__(message)
        self.output = output


class ToolGuardrailTripwireTriggered(RuntimeError):
    """
    Not raised by the framework itself — a tripped tool guardrail degrades
    gracefully by default (see run_tool_input_guardrails/run_tool_output_guardrails),
    the same way ordinary tool errors already do. Raise this yourself from inside
    a tool guardrail function if you want a trip to hard-abort the whole run
    instead of letting the model react to a substituted message and continue.
    """
    def __init__(self, message: str, output: GuardrailFunctionOutput):
        super().__init__(message)
        self.output = output


_TRIPWIRE_EXCEPTIONS = (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ToolGuardrailTripwireTriggered,
)

TInput = TypeVar("TInput")


@dataclass
class _Guardrail(Generic[TInput]):
    """Wraps a guardrail function (sync or async)."""
    fn: Callable[..., Any]
    name: str

    def run(self, *args, **kwargs) -> GuardrailFunctionOutput:
        """Execute guardrail function; supports sync/async via asyncio.run()."""
        try:
            if inspect.iscoroutinefunction(self.fn):
                try:
                    return asyncio.run(self.fn(*args, **kwargs))
                except RuntimeError:
                    # already inside a loop (e.g., Jupyter) → fallback
                    import nest_asyncio  # type: ignore
                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(self.fn(*args, **kwargs))
            else:
                return self.fn(*args, **kwargs)
        except _TRIPWIRE_EXCEPTIONS:
            # A guardrail deliberately raising one of the SDK's own tripwire
            # exceptions (the documented hard-stop recipe for tool guardrails)
            # must propagate as-is, not get downgraded to a generic RuntimeError
            # below — otherwise `except ToolGuardrailTripwireTriggered:` could
            # never work for a caller.
            raise
        except Exception as e:
            raise RuntimeError(f"Guardrail '{self.name}' raised an exception: {e}") from e


def input_guardrail(fn: Callable[..., Any]) -> _Guardrail[str]:
    """
    Decorator for input guardrails.

    Signature:
      def my_guardrail(ctx, agent, input: str | list[Any]) -> GuardrailFunctionOutput: ...
    """
    return _Guardrail(fn=fn, name=fn.__name__)


def output_guardrail(fn: Callable[..., Any]) -> _Guardrail[Any]:
    """
    Decorator for output guardrails.

    Signature:
      def my_guardrail(ctx, agent, output: Any) -> GuardrailFunctionOutput: ...
    """
    return _Guardrail(fn=fn, name=fn.__name__)


def tool_input_guardrail(fn: Callable[..., Any]) -> _Guardrail[Any]:
    """
    Decorator for tool input guardrails — run before a tool executes.

    Signature:
      def my_guardrail(ctx, agent, call: ToolCall, kwargs: dict) -> GuardrailFunctionOutput: ...

    `kwargs` are the tool's already-validated arguments (post `Tool.parse_args()`).
    A trip blocks the tool call and substitutes a message for the model instead
    of raising — see run_tool_input_guardrails.
    """
    return _Guardrail(fn=fn, name=fn.__name__)


def tool_output_guardrail(fn: Callable[..., Any]) -> _Guardrail[Any]:
    """
    Decorator for tool output guardrails — run after a tool returns, before its
    result reaches memory.

    Signature:
      def my_guardrail(ctx, agent, call: ToolCall, kwargs: dict, output: str) -> GuardrailFunctionOutput: ...
    """
    return _Guardrail(fn=fn, name=fn.__name__)


def run_input_guardrails(
    *,
    guards: Sequence[_Guardrail[str]],
    ctx: Any,     # RunContextWrapper[None]
    agent: Any,   # Agent (kept as Any to avoid runtime import)
    user_input: str,
) -> None:
    for g in guards:
        out = g.run(ctx, agent, user_input)
        if not isinstance(out, GuardrailFunctionOutput):
            raise TypeError(
                f"Input guardrail '{g.name}' must return GuardrailFunctionOutput, got {type(out)}"
            )
        if out.tripwire_triggered:
            raise InputGuardrailTripwireTriggered(
                f"Input guardrail '{g.name}' tripwire triggered.", out
            )


def run_output_guardrails(
    *,
    guards: Sequence[_Guardrail[Any]],
    ctx: Any,     # RunContextWrapper[None]
    agent: Any,   # Agent
    final_output: Any,
) -> None:
    for g in guards:
        out = g.run(ctx, agent, final_output)
        if not isinstance(out, GuardrailFunctionOutput):
            raise TypeError(
                f"Output guardrail '{g.name}' must return GuardrailFunctionOutput, got {type(out)}"
            )
        if out.tripwire_triggered:
            raise OutputGuardrailTripwireTriggered(
                f"Output guardrail '{g.name}' tripwire triggered.", out
            )


def run_tool_input_guardrails(
    *,
    guards: Sequence[_Guardrail[Any]],
    ctx: Any,           # RunContextWrapper[None]
    agent: Any,          # Agent
    call: "ToolCall",
    kwargs: Dict[str, Any],
) -> Optional[str]:
    """
    Returns None if no guardrail tripped. Returns a substitution string on the
    first trip — the caller should use it as the tool's result without ever
    invoking tool.run().
    """
    for g in guards:
        out = g.run(ctx, agent, call, kwargs)
        if not isinstance(out, GuardrailFunctionOutput):
            raise TypeError(
                f"Tool input guardrail '{g.name}' must return GuardrailFunctionOutput, got {type(out)}"
            )
        if out.tripwire_triggered:
            reason = out.reason or f"'{g.name}' blocked this tool call."
            return f"[Tool Guardrail Blocked] {reason}"
    return None


def run_tool_output_guardrails(
    *,
    guards: Sequence[_Guardrail[Any]],
    ctx: Any,           # RunContextWrapper[None]
    agent: Any,          # Agent
    call: "ToolCall",
    kwargs: Dict[str, Any],
    output: str,
) -> Optional[str]:
    """
    Returns None if no guardrail tripped. Returns a substitution string on the
    first trip — the caller should use it in place of the tool's real output.
    """
    for g in guards:
        out = g.run(ctx, agent, call, kwargs, output)
        if not isinstance(out, GuardrailFunctionOutput):
            raise TypeError(
                f"Tool output guardrail '{g.name}' must return GuardrailFunctionOutput, got {type(out)}"
            )
        if out.tripwire_triggered:
            reason = out.reason or f"'{g.name}' blocked this tool result."
            return f"[Tool Guardrail Blocked] {reason}"
    return None
