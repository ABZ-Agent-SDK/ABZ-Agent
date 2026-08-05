# abzagent/utils/logging.py
from __future__ import annotations
from rich.panel import Panel
from rich.json import JSON
from rich.console import Console

_console = Console()

def log_step(title: str, payload: str) -> None:
    try:
        # pretty print JSON if possible
        j = JSON(payload)
        _console.print(Panel(j, title=title))
    except Exception:
        _console.print(Panel(payload, title=title))
