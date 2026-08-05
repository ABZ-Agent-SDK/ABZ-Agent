# abzagent/extensions/handoffs_filter.py
from __future__ import annotations
from typing import Callable

from ..core.handoffs import HandoffInputData


def remove_all_tools(data: HandoffInputData) -> HandoffInputData:
    """Drop tool messages from history before handoff."""
    return HandoffInputData(messages=[m for m in data.messages if m.role != "tool"])


def keep_last_n_turns(n: int) -> Callable[[HandoffInputData], HandoffInputData]:
    """Keep only the last n messages (approximate turn packing)."""
    def _fn(data: HandoffInputData) -> HandoffInputData:
        return HandoffInputData(messages=data.messages[-2 * n:])
    return _fn
