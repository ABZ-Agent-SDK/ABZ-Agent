# abzagent/extensions/handoff_prompt.py
# Single source of truth lives in core/handoffs.py (Agent._build_prompt() auto-splices
# this whenever an agent has handoffs configured — this re-export is for anyone who
# wants to prepend it to instructions manually instead).
from ..core.handoffs import RECOMMENDED_PROMPT_PREFIX  # noqa: F401


def prompt_with_handoff_instructions(prompt: str) -> str:
    return f"{RECOMMENDED_PROMPT_PREFIX}\n{prompt}"
