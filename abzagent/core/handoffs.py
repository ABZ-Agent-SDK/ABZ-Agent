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
You are a ROUTER first. Before answering anything, check whether one of your \
transfer_to_<agent_name_slug> tools is a clear match for what the user is asking.

- If a specialist tool clearly matches this specific request: call it immediately. \
Do not draft an answer first, do not partially answer, do not explain that you're \
transferring, and do not ask the user for permission or confirmation first — just call \
the tool, right now, in this turn.
- If more than one specialist could plausibly match, pick the single best match yourself. \
Do not ask the user to choose between them.
- If no specialist tool matches this specific request (e.g. small talk, or a question \
none of your specialists cover): answer normally yourself.
- A transfer is a complete, final action. Never call a transfer tool and then also \
answer the question yourself in the same turn.

Return ONLY a JSON object for the tool call, e.g.:

{"tool":"transfer_to_refund_agent","args":{"message":"Customer is asking for a refund for order #123."}}
"""

# Used only in native-tool-calling mode (Agent._build_prompt(), gated on
# provider.supports_native_tools). The provider's own tool-calling mechanism
# already handles HOW to invoke a transfer; this only needs to teach the
# model WHEN to — no JSON-formatting instructions, unlike
# RECOMMENDED_PROMPT_PREFIX above, which stays exactly as-is (content aside)
# since it's publicly re-exported via extensions/handoff_prompt.py.
RECOMMENDED_PROMPT_PREFIX_NATIVE = """\
You are a ROUTER first. Before answering anything, check whether one of your \
transfer tools is a clear match for what the user is asking.

- If a specialist tool clearly matches this specific request: call it immediately. \
Do not draft an answer first, do not partially answer, do not explain that you're \
transferring, and do not ask the user for permission or confirmation first — just call \
the tool, right now, in this turn.
- If more than one specialist could plausibly match, pick the single best match yourself. \
Do not ask the user to choose between them.
- If no specialist tool matches this specific request (e.g. small talk, or a question \
none of your specialists cover): answer normally yourself.
- A transfer is a complete, final action. Never call a transfer tool and then also \
answer the question yourself in the same turn.
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
        self.tool_description = tool_description_override or _auto_description(target_agent)

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


# Caps the auto-derived description's length — native-mode tool calling sends
# every tool's full description as part of its JSON schema on every turn, so an
# unbounded copy of a target agent's (possibly long) instructions would bloat
# every request, not just the one where the handoff is relevant.
_MAX_AUTO_DESCRIPTION_INSTRUCTIONS_CHARS = 300


def _strip_redundant_persona_prefix(text: str, agent_name: str) -> str:
    """Strip a leading "You are <Name>[.,]" clause from instructions before it's
    embedded into a tool description — `_auto_description` already restates the
    agent's name via "Transfer to the '<Name>' agent.", so echoing a persona
    clause with the same name again right after reads redundant."""
    pattern = rf"^you\s+are\s+(the\s+)?{re.escape(agent_name)}\b[.,]?\s*"
    stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    return stripped or text


def _truncate_description_text(text: str, limit: int) -> str:
    """Truncate at the last word boundary at or before `limit`, so a long
    instructions string never gets cut off mid-word inside a tool description."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    space = truncated.rfind(" ")
    if space > 0:
        truncated = truncated[:space]
    return truncated.rstrip() + "..."


def _auto_description(target_agent: "Agent") -> str:
    """
    Default tool description for a handoff, with no override given. Auto-derives
    a "when to use this" hint from the target agent's own instructions — the
    same text the developer already had to write, no extra config — so a bare
    Agent(name=..., instructions=...) in handoffs=[...] gets a genuinely useful
    routing description automatically. Falls back to a generic description when
    instructions are dynamic (a callable) and can't be safely introspected
    without actually calling it.
    """
    instructions_src = getattr(target_agent, "_instructions_src", None)
    if isinstance(instructions_src, str):
        text = _strip_redundant_persona_prefix(instructions_src.strip(), target_agent.name)
        if text:
            text = _truncate_description_text(text, _MAX_AUTO_DESCRIPTION_INSTRUCTIONS_CHARS)
            return f"Transfer to the '{target_agent.name}' agent. Use this when the request matches: {text}"

    return f"Transfer the conversation to the '{target_agent.name}' agent."
