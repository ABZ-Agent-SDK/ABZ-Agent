# abzagent/extensions/handoffs_filter.py
from __future__ import annotations
from typing import List, Dict
from ..core.handoffs import HandoffInputData

def remove_all_tools(data: HandoffInputData) -> HandoffInputData:
    """Drop tool messages from history before handoff."""
    filtered = [m for m in data.history if m.get("role") != "tool"]
    return HandoffInputData(user_message=data.user_message, history=filtered, metadata=data.metadata)

def keep_last_n_turns(n: int):
    """Keep the last n user/assistant turns only."""
    def _fn(data: HandoffInputData) -> HandoffInputData:
        out: List[Dict[str, str]] = []
        # simple slice of last 2n messages (approximate turn packing)
        out = data.history[-2*n:]
        return HandoffInputData(user_message=data.user_message, history=out, metadata=data.metadata)
    return _fn
