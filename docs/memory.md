# Memory

`Memory` gives an `Agent` a running transcript of the conversation so far, which is
replayed as plain-text context on every call to `agent.run()`. It is deliberately minimal:
an in-process list of messages, with no persistence, no database, and no vector search.

> From the source docstring: *"Simple conversation buffer memory. Replace with
> vector/redis/etc as needed."* — `Memory` is a small reference implementation meant to be
> subclassed or swapped out, not a batteries-included persistence layer. If you need
> durability across process restarts, semantic search over history, or multi-session
> storage, you currently need to implement it yourself (see [Extending Memory](#extending-memory)).

## Import

```python
from abzagent import Memory
```

## Quick start

```python
from abzagent import Agent, Memory

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    memory=Memory(),
)

print(agent.run("My name is Abu.").content)
print(agent.run("What's my name?").content)   # remembers the prior turn
```

If you don't pass `memory=` explicitly, `Agent.__init__` creates a fresh `Memory()` for you
automatically (`self.memory = memory or Memory()`), so a bare `Agent(...)` still has
working short-term memory by default.

## `Memory` class

```python
class Memory:
    def __init__(self)
    def remember(self, role: str, content: str) -> None
    def load(self) -> Iterable[Message]
    def to_prompt(self) -> str
```

- `Memory()` takes **no constructor arguments** — it always starts empty, backed by a new `MessageBuffer`.
- `remember(role, content)` — appends a message to the buffer. `role` is a free-form string in practice, though the type alias used throughout the SDK is `Literal["system", "user", "assistant", "tool"]`.
- `load()` — returns all stored messages as a `list[Message]`, in insertion order.
- `to_prompt()` — renders the entire buffer as a plain-text chat transcript (see format below). This is what `Agent._build_prompt()` calls internally to inject history into every LLM request.

```python
memory = Memory()
memory.remember("user", "Hello")
messages = memory.load()          # [Message(role="user", content="Hello")]
prompt_text = memory.to_prompt()  # "[USER]: Hello"
```

## `Message` and `MessageBuffer`

`Memory` is a thin wrapper around these two lower-level types
(`abzagent.core.messages`):

```python
Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str

class MessageBuffer:
    def add(self, role: Role, content: str) -> None
    def to_prompt(self) -> str
    def __len__(self) -> int
    def __iter__(self) -> Iterator[Message]
```

`MessageBuffer.to_prompt()` renders each message as `[ROLE]: content` (role upper-cased),
joined by blank lines:

```
[USER]: Hello

[TOOL]: 4
```

This flat string — not a structured multi-message payload — is what actually gets sent to
the Gemini/Groq provider on every call, concatenated after the system prompt and before the
current user message. Neither provider's native multi-turn message array format is used.

## How `Agent.run()` uses Memory — important behavioral detail

Inside `run()`, the SDK calls `self.memory.remember(...)` in exactly two places:

```python
self.memory.remember("user", user_message)   # every call, right after guardrails
...
self.memory.remember("tool", obs)             # only when a tool call was executed
```

**The model's own final text answers are never written back into `Memory`.** There is no
`self.memory.remember("assistant", ...)` call anywhere in `agent.py`. This means:

- Only user messages and tool-execution results persist across turns.
- If an agent replies with plain text (no tool call), that reply is returned to the caller
  in `AgentResult.content` but is **not** added to memory — so the agent will not "recall"
  its own previous answer on the next turn, only what the user said and what tools returned.
- If your use case depends on the agent remembering its own prior responses, you currently
  need to call `agent.memory.remember("assistant", result.content)` yourself after each
  `run()` call, or subclass `Memory` to do this automatically.

```python
result = agent.run("What is 2 + 2?")
agent.memory.remember("assistant", result.content)   # manual workaround
```

## Memory is not session-scoped

`Memory` has no concept of conversation/session IDs — it is just a plain object holding a
list. There is no `SessionManager`, `SessionState`, or similar construct anywhere in the
SDK. Practical implications:

- **One conversation per `Memory` instance.** To run multiple independent conversations
  concurrently (e.g. one per end user), create a separate `Memory()` per conversation and
  either construct a separate `Agent` per conversation or swap `agent.memory` between calls.
- **Nothing expires or trims automatically.** The buffer grows for as long as the process
  keeps calling `remember()`; there is no built-in max-length, token-budget trimming, or
  summarization. For long-running agents you may want to periodically inspect
  `len(memory.buffer)` / `memory.load()` and prune or summarize old messages yourself.
- **Nothing is persisted to disk or a database.** Restarting the process loses all memory.

```python
# One Memory per user/session
sessions: dict[str, Memory] = {}

def get_agent_for(user_id: str) -> Agent:
    if user_id not in sessions:
        sessions[user_id] = Memory()
    return Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        memory=sessions[user_id],
    )
```

## Extending Memory

Because `Memory` is intentionally minimal, the documented (docstring-endorsed) path for
adding persistence, vector search, or a database backend is to implement your own class
with the same interface and pass it in via `Agent(memory=...)`. `Agent` only calls
`remember()`, `to_prompt()` (indirectly, through `Memory`), and reads `self.memory` — any
object exposing a compatible `remember(role, content)` / `to_prompt()` surface will work as
a drop-in replacement, e.g.:

```python
from abzagent.core.messages import MessageBuffer

class RedisMemory:
    def __init__(self, redis_client, key: str):
        self._redis = redis_client
        self._key = key

    def remember(self, role: str, content: str) -> None:
        self._redis.rpush(self._key, f"{role}:{content}")

    def load(self):
        return [entry.split(":", 1) for entry in self._redis.lrange(self._key, 0, -1)]

    def to_prompt(self) -> str:
        return "\n\n".join(
            f"[{role.upper()}]: {content}" for role, content in self.load()
        )

agent = Agent(name="Assistant", instructions="...", memory=RedisMemory(client, "user:42"))
```

There is no built-in vector-store or Redis backend shipped with the SDK today — this
pattern is the intended extension point, not a documented feature you can import off the
shelf.

## Known limitations summary

| Behavior | Status |
|---|---|
| Persistence across process restarts | Not implemented — in-memory only |
| Vector / semantic search over history | Not implemented |
| Assistant replies written back to memory | Not automatic — `run()` never calls `remember("assistant", ...)` |
| Session/conversation scoping | Not implemented — one `Memory` instance = one conversation |
| Automatic trimming / token-budget management | Not implemented — buffer grows unbounded |
