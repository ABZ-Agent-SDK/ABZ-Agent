# abzagent/core/__init__.py
from __future__ import annotations
from .agent import Agent, AgentResult  # noqa: F401
from .memory import Memory  # noqa: F401
from .tools import Tool, ToolCall, ToolSchema, tool_to_schema, FunctionTool, function_tool  # noqa: F401
from .output import AgentOutputSchema, ModelBehaviorError  # noqa: F401
from .context import RunContextWrapper  # noqa: F401
from ..providers.base import GenerationResult, ModelProvider  # noqa: F401

# Optional re-exports (guard if not present to avoid circulars during partial installs)
try:
    from .handoffs import (  # noqa: F401
        Handoff,
        handoff,
        HandoffError,
        CircularHandoffError,
        MaxHandoffDepthExceededError,
        HandoffInputData,
    )
except Exception:
    Handoff = handoff = None  # type: ignore
    HandoffError = CircularHandoffError = MaxHandoffDepthExceededError = None  # type: ignore
    HandoffInputData = None  # type: ignore

try:
    from .guardrails import (  # noqa: F401
        input_guardrail,
        output_guardrail,
        tool_input_guardrail,
        tool_output_guardrail,
        GuardrailFunctionOutput,
        InputGuardrailTripwireTriggered,
        OutputGuardrailTripwireTriggered,
        ToolGuardrailTripwireTriggered,
    )
except Exception:
    input_guardrail = output_guardrail = GuardrailFunctionOutput = None  # type: ignore
    tool_input_guardrail = tool_output_guardrail = None  # type: ignore
    InputGuardrailTripwireTriggered = OutputGuardrailTripwireTriggered = None  # type: ignore
    ToolGuardrailTripwireTriggered = None  # type: ignore

__all__ = [
    # Core
    "Agent", "AgentResult", "Memory",
    # Tools
    "Tool", "ToolCall", "ToolSchema", "tool_to_schema", "FunctionTool", "function_tool",
    # Structured output
    "AgentOutputSchema", "ModelBehaviorError",
    # Handoffs
    "Handoff", "handoff", "RunContextWrapper",
    "HandoffError", "CircularHandoffError", "MaxHandoffDepthExceededError", "HandoffInputData",
    # Providers
    "GenerationResult", "ModelProvider",
    # Optional
    "input_guardrail", "output_guardrail",
    "tool_input_guardrail", "tool_output_guardrail",
    "GuardrailFunctionOutput",
    "InputGuardrailTripwireTriggered", "OutputGuardrailTripwireTriggered",
    "ToolGuardrailTripwireTriggered",
]
