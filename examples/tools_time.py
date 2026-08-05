from datetime import datetime
import zoneinfo
from abzagent import function_tool

@function_tool()
def current_time(timezone: str = "UTC") -> str:
    """Get the current time in a given IANA timezone.

    Args:
        timezone: e.g. 'America/Los_Angeles', 'Asia/Karachi'
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except Exception:
        return "Invalid timezone."
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
