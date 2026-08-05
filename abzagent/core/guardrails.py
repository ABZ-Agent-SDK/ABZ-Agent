# abzagent/core/guardrails.py
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

# ⚠️ Do NOT import Agent at runtime — that creates a circular import with core.agent.
if TYPE_CHECKING:
    from .agent import Agent  # type hint only; not executed at import time


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
    # raise after loop if any tripped (stop at first for clarity)
    for g in guards:
        out = g.run(ctx, agent, final_output)
        if out.tripwire_triggered:
            raise OutputGuardrailTripwireTriggered(
                f"Output guardrail '{g.name}' tripwire triggered.", out
            )
