from .messages import MessageBuffer, Message
from typing import Iterable


class Memory:
    """Simple conversation buffer memory.
    Replace with vector/redis/etc as needed.
    """

    def __init__(self):
        self.buffer = MessageBuffer()

    def remember(self, role: str, content: str):
        self.buffer.add(role, content)

    def load(self) -> Iterable[Message]:
        return list(self.buffer)

    def to_prompt(self) -> str:
        return self.buffer.to_prompt()