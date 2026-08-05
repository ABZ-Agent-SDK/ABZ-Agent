from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from abzagent.core.tools import Tool
from pydantic import BaseModel

class ClockSchema(BaseModel):
    timezone: str | None = None  # e.g., "America/Los_Angeles"

class TimeTool(Tool):
    name = "clock"  # 👈 match model’s call
    description = "Return current ISO timestamp for an optional IANA timezone."
    schema = ClockSchema

    def run(self, **kwargs) -> str:
        tz = kwargs.get("timezone") or "UTC"
        try:
            now = datetime.now(ZoneInfo(tz))
        except Exception:
            now = datetime.now(timezone.utc)
        return now.isoformat()
