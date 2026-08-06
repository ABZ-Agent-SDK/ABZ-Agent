import os
import logging

# --- Silence Gemini / gRPC / absl warnings globally ---

# must be set before grpc or google libs load
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_SUPPRESS_LOGS"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# silence python warnings too
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# abzagent/__init__.py
"""
ABZ Agent SDK — Simplify building AI Agents with Google Gemini.

Public API:
    from abzagent import Agent, Memory, function_tool
"""


# ─────────────────────────────────────────────
# Silence gRPC / absl / TensorFlow warnings early



# ─────────────────────────────────────────────
# Core public imports
# ─────────────────────────────────────────────
from .core.agent import Agent, AgentResult
from .core.memory import Memory
from .core.tools import Tool, ToolCall, ToolSchema, tool_to_schema, function_tool
from .core.output import AgentOutputSchema, ModelBehaviorError
from .core.context import RunContextWrapper
from .providers.base import GenerationResult, ModelProvider
from .core.handoffs import (
    Handoff,
    handoff,
    HandoffError,
    CircularHandoffError,
    MaxHandoffDepthExceededError,
    InvalidHandoffTargetError,
    HandoffInputData,
)
from .core.guardrails import (
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ToolGuardrailTripwireTriggered,
)

# Optional: expose SDK config and providers
from .config import SDKConfig
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider


# ─────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────
__all__ = [
    "Agent",
    "AgentResult",
    "Memory",
    "Tool",
    "ToolCall",
    "ToolSchema",
    "tool_to_schema",
    "function_tool",
    "AgentOutputSchema",
    "ModelBehaviorError",
    "RunContextWrapper",
    "GenerationResult",
    "ModelProvider",
    "Handoff",
    "handoff",
    "HandoffError",
    "CircularHandoffError",
    "MaxHandoffDepthExceededError",
    "InvalidHandoffTargetError",
    "HandoffInputData",
    "GuardrailFunctionOutput",
    "input_guardrail",
    "output_guardrail",
    "tool_input_guardrail",
    "tool_output_guardrail",
    "InputGuardrailTripwireTriggered",
    "OutputGuardrailTripwireTriggered",
    "ToolGuardrailTripwireTriggered",
    "SDKConfig",
    "GeminiProvider",
    "GroqProvider",
]

__version__ = "0.5.1"
__author__ = "Abu Bakar"
__license__ = "MIT"

# ─────────────────────────────────────────────
# Friendly startup banner (optional)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"ABZ Agent SDK v{__version__} — Build AI Agents with Gemini & Groq")
