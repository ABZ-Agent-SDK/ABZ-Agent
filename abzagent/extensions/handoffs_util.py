# abzagent/extensions/handoffs_util.py
from __future__ import annotations
from typing import Optional, Tuple

HANDOFF_MARK_PREFIX = "<<<HANDOFF:"

def parse_handoff_output(text: str) -> Tuple[Optional[str], str]:
    """
    If text starts with '<<<HANDOFF:<Agent Name>>>', return (agent_name, cleaned_text).
    Otherwise return (None, text).
    """
    if text.startswith(HANDOFF_MARK_PREFIX):
        try:
            first_line, *rest = text.splitlines()
            agent_name = first_line[len(HANDOFF_MARK_PREFIX):].removesuffix(">>>").strip()
            cleaned = "\n".join(rest).lstrip()
            return agent_name, cleaned
        except Exception:
            return None, text
    return None, text
