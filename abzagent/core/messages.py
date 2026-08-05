from dataclasses import dataclass
from typing import Literal, List


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str


class MessageBuffer:
    def __init__(self):
        self._messages: List[Message] = []

    def add(self, role: Role, content: str):
        self._messages.append(Message(role, content))

    def to_prompt(self) -> str:
        # Minimal prompt compiler: render as chat transcript
        lines = []
        for m in self._messages:
            lines.append(f"[{m.role.upper()}]: {m.content}")
        return "\n\n".join(lines)

    def __len__(self):
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)