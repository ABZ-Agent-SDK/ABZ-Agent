from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
from dataclasses import asdict

from .abz_handoff_core import (
    SessionManager,
    ContextBridge,
    HandoffController as BaseHandoffController,
    HandoffRequest,
    SessionContext
)
from .tools import Tool, ToolCall

# ---------------- Handoff Wrapper ----------------
class Handoff:
    """Wraps an agent or agent name to be used as a handoff tool."""

    def __init__(self, target_agent: Any, session_manager: Optional[SessionManager] = None):
        """
        target_agent: can be an Agent instance, agent name string, or any identifier
        session_manager: optional custom session manager
        """
        self.target_agent = target_agent
        self.session_manager = session_manager or SessionManager()
        self.controller = BaseHandoffController(self.session_manager)

    def to_tool(self, current_agent: Any) -> Tool:
        """Return a Tool object that calls this handoff."""
        target_name = getattr(self.target_agent, "name", str(self.target_agent))

        class _HandoffTool(Tool):
            name = f"handoff_to_{target_name}"
            description = f"Handoff tool from {getattr(current_agent, 'name', 'Agent')} to {target_name}"

            schema = None  # free-form message

            def run(self, **kwargs) -> str:
                user_message = kwargs.get("message", "")
                # Build handoff request
                # session_payload must be provided; we simulate by serializing current memory
                session_ctx: SessionContext = getattr(current_agent, "memory", None)
                if not session_ctx:
                    # fallback: create a dummy session
                    session_ctx = self._create_dummy_session(user_message)

                payload = ContextBridge.serialize(session_ctx)

                req = HandoffRequest(
                    from_agent=getattr(current_agent, "name", None),
                    to_agent=target_name,
                    session_payload=payload,
                    reason="handoff triggered by agent",
                )

                # Run async controller in sync context
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()

                return loop.run_until_complete(self.controller.start_handoff(req))["message"]

            def _create_dummy_session(self, user_message: str) -> SessionContext:
                # fallback dummy session for memory-less agent
                from .memory import Memory  # optional; if needed
                s = SessionContext(session_id="dummy")
                s.append_message("user", user_message)
                return s

        return _HandoffTool()


# ---------------- Public handoff() factory ----------------
def handoff(agent: Any) -> Handoff:
    """
    Wraps a target agent (or agent name) as a Handoff object for ABZ Agent SDK.
    Use in Agent definition:

        sales_agent = Agent(...)
        support_agent = Agent(..., handoffs=[handoff(sales_agent)])
    """
    return Handoff(agent)
