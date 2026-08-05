# abzagent/core/context.py
from __future__ import annotations


class RunContextWrapper:
    """
    Passed to dynamic instruction functions, guardrails, and handoff callbacks.
    Extracted into its own module so both agent.py and handoffs.py can import
    it unconditionally without a circular-import shape.
    """

    def __init__(self, current_agent=None, target_agent=None, memory=None, steps=None, context=None):
        self.current_agent = current_agent
        self.target_agent = target_agent
        self.memory = memory
        self.steps = steps or []
        self.context = context
