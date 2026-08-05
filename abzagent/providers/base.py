# abzagent/providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from ..core.tools import ToolCall, ToolSchema

if TYPE_CHECKING:
    from ..core.output import AgentOutputSchema


@dataclass
class GenerationResult:
    """
    Result of a single provider.generate() call.

    Exactly one of `text` / `tool_calls` is meaningful in practice: either the
    model produced a final (or intermediate) text response, or it invoked one
    or more tools. When multiple tool calls come back in one response (both
    Gemini and Groq can return a list), only `tool_calls[0]` is acted on —
    that policy lives in Agent, not here or per-provider.
    """
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert not (self.text and self.tool_calls), (
            "GenerationResult should not have both text and tool_calls set."
        )


class ModelProvider(ABC):
    # Declarative capability flag — no runtime per-model probing. Providers that
    # support native function/tool calling set this True (the default) and
    # translate `tools=` into their own request/response shape. A hypothetical
    # future provider without tool-calling support sets this False, and Agent
    # falls back to the prompt-text + JSON-blob-parsing convention instead.
    supports_native_tools: bool = True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        tools: Optional[List[ToolSchema]] = None,
        output_schema: Optional["AgentOutputSchema"] = None,
        strict: bool = True,
    ) -> GenerationResult:
        """
        Generate from the model, given one flat prompt string.

        `tools`, when given and `supports_native_tools` is True, should be
        translated into the provider's native function/tool-calling request;
        any tool calls the model makes come back via
        `GenerationResult.tool_calls`, not embedded in `.text`.

        `output_schema`, when given, means the Agent expects the final answer
        to be JSON matching that schema — providers that support a native
        structured-output mode should use it. `strict=False` means tools are
        active this turn, so providers must not lock generation to the exact
        output schema (a tool call may legitimately happen instead); a looser
        "valid JSON" mode should be used if available. `tools` and schema-locked
        structured output should never both be requested in the same call.
        """
        ...
