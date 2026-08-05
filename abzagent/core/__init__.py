# abzagent/core/__init__.py
from __future__ import annotations
from .agent import Agent, AgentResult  # noqa: F401
from .memory import Memory  # noqa: F401
from .tools import Tool, ToolCall, FunctionTool, function_tool  # noqa: F401
from .output import AgentOutputSchema, ModelBehaviorError  # noqa: F401

# Optional re-exports (guard if not present to avoid circulars during partial installs)
try:
    from .handoffs import Handoff, handoff, RunContextWrapper  # noqa: F401
except Exception:
    Handoff = handoff = RunContextWrapper = None  # type: ignore

try:
    from .guardrails import (  # noqa: F401
        input_guardrail,
        output_guardrail,
        GuardrailFunctionOutput,
        InputGuardrailTripwireTriggered,
        OutputGuardrailTripwireTriggered,
    )
except Exception:
    input_guardrail = output_guardrail = GuardrailFunctionOutput = None  # type: ignore
    InputGuardrailTripwireTriggered = OutputGuardrailTripwireTriggered = None  # type: ignore

__all__ = [
    # Core
    "Agent", "AgentResult", "Memory",
    # Tools
    "Tool", "ToolCall", "FunctionTool", "function_tool",
    # Structured output
    "AgentOutputSchema", "ModelBehaviorError",
    # Optional
    "Handoff", "handoff", "RunContextWrapper",
    "input_guardrail", "output_guardrail",
    "GuardrailFunctionOutput",
    "InputGuardrailTripwireTriggered", "OutputGuardrailTripwireTriggered",
]
