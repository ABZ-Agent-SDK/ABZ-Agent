"""
Structured Output example.

Define a Pydantic model as `output_type` and get a real Python object back —
no manual JSON prompting, no manual parsing.
"""

from typing import List
from pydantic import BaseModel
from abzagent import Agent


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]


agent = Agent(
    name="Event Extractor",
    instructions="Extract the calendar event described by the user.",
    model="gemini-2.0-flash",  # swap for a Groq model — same code
    output_type=CalendarEvent,
)

result = agent.run("Standup tomorrow (2026-08-05) with Abu and Sara.")

print(result.parsed.name)          # "Standup"
print(result.parsed.participants)  # ["Abu", "Sara"]
