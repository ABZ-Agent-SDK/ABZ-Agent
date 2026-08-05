# abzagent/core/handoffs.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING

from pydantic import BaseModel

from .context import RunContextWrapper
from .messages import Message
from .tools import Tool

# ⚠️ Do NOT import Agent at runtime — that creates a circular import with core.agent.
if TYPE_CHECKING:
    from .agent import Agent


RECOMMENDED_PROMPT_PREFIX = """\
You can DELEGATE tasks to other specialized agents using tools named like:
- transfer_to_<agent_name_slug>

When you decide another agent is better suited, call the correct transfer tool ONCE with any minimal context it needs.
Return ONLY a JSON object for the tool call, e.g.:

{"tool":"transfer_to_refund_agent","args":{"message":"Customer is asking for a refund for order #123."}}

After delegating, do not repeat the answer yourself.
"""

# Used only in native-tool-calling mode (Agent._build_prompt(), gated on
# provider.supports_native_tools). The provider's own tool-calling mechanism
# already handles HOW to invoke a transfer; this only needs to teach the
# model WHEN to — no JSON-formatting instructions, unlike
# RECOMMENDED_PROMPT_PREFIX above, which stays exactly as-is since it's
# publicly re-exported via extensions/handoff_prompt.py.
RECOMMENDED_PROMPT_PREFIX_NATIVE = """\
You can delegate this conversation to another, more specialized agent by \
calling the appropriate transfer tool. Do this when a specialized agent is \
clearly better suited to handle the request than you are. After delegating, \
do not also answer yourself — the transfer is the complete action.
"""

MAX_HANDOFF_DEPTH = 5


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class HandoffError(RuntimeError):
    """Base class for structural handoff failures (same category as guardrail
    tripwires — an aborted run, not a garden-variety model/tool-arg mistake)."""


class InvalidHandoffTargetError(HandoffError):
    """Raised eagerly when a handoff target doesn't look like an Agent."""


class CircularHandoffError(HandoffError):
    """Raised when a handoff chain would revisit an agent already in it."""


class MaxHandoffDepthExceededError(HandoffError):
    """Raised when a handoff chain exceeds MAX_HANDOFF_DEPTH."""


# ──────────────────────────────────────────────────────────────────────────────
# Input data / filters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HandoffInputData:
    """The conversation history about to be handed to the next agent."""
    messages: List[Message] = field(default_factory=list)


InputFilter = Callable[[HandoffInputData], HandoffInputData]


class _DefaultHandoffArgs(BaseModel):
    message: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Handoff
# ──────────────────────────────────────────────────────────────────────────────

class Handoff:
    """Wraps a target Agent so it can be offered to another Agent as a
    transfer_to_<name> tool. Construct via the handoff() factory below."""

    def __init__(
        self,
        target_agent: "Agent",
        *,
        tool_name_override: Optional[str] = None,
        tool_description_override: Optional[str] = None,
        input_filter: Optional[InputFilter] = None,
        on_handoff: Optional[Callable[..., Any]] = None,
        input_type: Optional[Type[BaseModel]] = None,
    ) -> None:
        if not hasattr(target_agent, "run") or not hasattr(target_agent, "name"):
            raise InvalidHandoffTargetError(
                f"Handoff target must be an Agent (needs .name and .run()), got {target_agent!r}."
            )

        self.target_agent = target_agent
        self.input_filter = input_filter
        self.on_handoff = on_handoff
        self.input_type = input_type

        slug = _slugify(target_agent.name)
        self.tool_name = tool_name_override or f"transfer_to_{slug}"
        self.tool_description = (
            tool_description_override
            or f"Transfer the conversation to the '{target_agent.name}' agent."
        )

    def to_tool(self, source_agent: "Agent") -> Tool:
        handoff_obj = self

        class _HandoffTool(Tool):
            name = handoff_obj.tool_name
            description = handoff_obj.tool_description
            schema = handoff_obj.input_type or _DefaultHandoffArgs

            # Detection markers — Agent._run() checks these to intercept the
            # call before normal tool execution.
            is_handoff = True
            handoff = handoff_obj

            def run(self, **kwargs) -> str:  # pragma: no cover - never actually invoked
                # Agent._run() intercepts handoff tool calls before this would
                # be reached; kept only so direct/introspective calls fail safely.
                return f"[Handoff] This call should have been intercepted by Agent._run() for a transfer to '{handoff_obj.target_agent.name}'."

        return _HandoffTool()


def handoff(
    agent: "Agent",
    *,
    tool_name_override: Optional[str] = None,
    tool_description_override: Optional[str] = None,
    input_filter: Optional[InputFilter] = None,
    on_handoff: Optional[Callable[..., Any]] = None,
    input_type: Optional[Type[BaseModel]] = None,
) -> Handoff:
    """
    Wrap a target agent as a Handoff, for customization beyond the bare-Agent
    default (tool name/description overrides, input_filter, on_handoff, input_type):

        sales_agent = Agent(...)
        support_agent = Agent(..., handoffs=[handoff(sales_agent, input_filter=remove_all_tools)])

    For the common case, a bare Agent in `handoffs=[...]` is wrapped in a
    Handoff automatically — you don't need to call this yourself.
    """
    return Handoff(
        agent,
        tool_name_override=tool_name_override,
        tool_description_override=tool_description_override,
        input_filter=input_filter,
        on_handoff=on_handoff,
        input_type=input_type,
    )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "agent"
