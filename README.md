# ABZ Agent SDK

A lightweight Python SDK for building AI agents powered by **Google Gemini** and **Groq**. One consistent API, two providers, zero boilerplate.

```bash
pip install abz-agents
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Providers & Models](#providers--models)
- [The Agent](#the-agent)
- [AgentResult](#agentresult)
- [Memory](#memory)
- [Tools](#tools)
  - [function_tool decorator](#function_tool-decorator)
  - [Tool base class](#tool-base-class)
  - [FunctionTool (manual)](#functiontool-manual)
  - [Built-in Tools](#built-in-tools)
- [Multi-Step / Iterative Mode](#multi-step--iterative-mode)
- [Dynamic Instructions](#dynamic-instructions)
- [Structured Output](#structured-output)
- [Guardrails](#guardrails)
- [Handoffs](#handoffs)
- [Agent as a Tool](#agent-as-a-tool)
- [Verbose Mode](#verbose-mode)
- [CLI](#cli)
- [Environment Variables](#environment-variables)
- [Changelog](#changelog)

---

## Quick Start

```python
from dotenv import load_dotenv
load_dotenv()

from abzagent import Agent, Memory

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gemini-2.0-flash",
    memory=Memory(),
)

result = agent.run("What is the capital of France?")
print(result.content)
```

---

## Installation

```bash
pip install abz-agents
```

Create a `.env` file in your project root:

```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

The SDK loads `.env` automatically via `python-dotenv`. You do not need to call `load_dotenv()` manually, though it is harmless to do so.

---

## Providers & Models

The SDK supports **Google Gemini** and **Groq**. The provider is detected automatically from the model name — no extra configuration required.

### Gemini models

| Model | Notes |
|---|---|
| `gemini-2.0-flash` | Default. Fast and capable. |
| `gemini-2.0-flash-lite` | Lightest 2.x variant |
| `gemini-2.5-pro-exp-03-25` | Highest quality (experimental) |
| `gemini-1.5-pro` | Stable high-quality |
| `gemini-1.5-flash` | Stable fast |
| `gemini-1.5-flash-8b` | Fastest / smallest |

Requires `GEMINI_API_KEY`.

### Groq models

| Model | Notes |
|---|---|
| `qwen/qwen3-32b` | Balanced default for Groq |
| `qwen/qwen-2.5-72b-instruct` | High quality |
| `llama-3.3-70b-versatile` | Strong Llama 3.3 |
| `llama-3.1-8b-instant` | Fastest Llama |
| `mixtral-8x7b-32768` | Long context |
| `deepseek-r1-distill-llama-70b` | Reasoning |
| `gemma2-9b-it` | Compact |

Requires `GROQ_API_KEY`.

### Provider auto-detection

Any model name containing `qwen/`, `llama`, `mixtral`, `deepseek`, `gemma2`, or `gemma-` is automatically routed to Groq. Everything else goes to Gemini.

```python
# Gemini
agent = Agent(..., model="gemini-2.0-flash")

# Groq — detected automatically
agent = Agent(..., model="qwen/qwen3-32b")
agent = Agent(..., model="llama-3.3-70b-versatile")
```

---

## The Agent

```python
from abzagent import Agent, Memory

agent = Agent(
    name="My Agent",                        # required
    instructions="You are helpful.",        # required — string or function
    model="gemini-2.0-flash",               # optional, default: gemini-2.0-flash
    tools=[...],                            # optional list of Tool or plain functions
    handoffs=[...],                         # optional list of Handoff objects
    memory=Memory(),                        # optional, default: fresh Memory()
    verbose=False,                          # optional, print tool execution details
    max_iterations=1,                       # optional, default: 1 (single-turn)
    api_key="...",                          # optional, overrides env var
    output_type=None,                       # optional Pydantic model for structured output
    input_guardrails=[],                    # optional list of input guardrail functions
    output_guardrails=[],                   # optional list of output guardrail functions
)

result = agent.run("Hello!")
print(result.content)
```

### `agent.run(user_message, *, context=None)`

Runs the agent and returns an `AgentResult`. The optional `context` argument is forwarded to dynamic instructions functions and guardrails via `RunContextWrapper`.

### `agent.register_tool(tool)`

Add a tool after the agent has been created.

```python
agent.register_tool(my_tool)
```

---

## AgentResult

Every call to `agent.run()` returns an `AgentResult`:

| Attribute | Type | Description |
|---|---|---|
| `.content` | `str` | The final text response |
| `.steps` | `list[str]` | All intermediate model outputs and tool results |
| `.parsed` | `Any` | Typed Pydantic object when `output_type` is set |

```python
result = agent.run("What is 2 + 2?")
print(result.content)   # "4"
print(result.steps)     # list of raw model/tool outputs
```

---

## Memory

`Memory` is a simple conversation buffer that stores every turn and replays it as context on each call to `agent.run()`.

```python

from abzagent import Agent

agent = Agent(
    name="ABZ Helper",
    instructions="Be concise and use tools efficiently.",
    model="gemini-2.0-flash",
)

print(agent.run("What is 2 + 2?").content)

```

### Memory API

```python
memory = Memory()
memory.remember("user", "Hello")          # add a message
messages = memory.load()                   # list of Message objects
prompt_text = memory.to_prompt()           # render as chat transcript string
```

Each `Message` has `.role` (`user`, `assistant`, `tool`, `system`) and `.content`.

---

## Tools

Tools extend what an agent can do. The model decides when to call a tool by emitting a single-line JSON object:

```json
{"tool": "tool_name", "args": {"key": "value"}}
```

The SDK parses this, validates arguments with Pydantic, calls the tool, and feeds the result back to the model.

### function_tool decorator

The fastest way to create a tool. Decorate any Python function — the SDK reads its signature, type hints, and docstring to build the schema automatically.

```python
from abzagent import function_tool, Agent

@function_tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny, 24°C in {city}."

agent = Agent(
    name="WeatherBot",
    instructions="You help with weather questions.",
    model="gemini-2.0-flash",
    tools=[get_weather],
)

result = agent.run("What's the weather in Istanbul?")
print(result.content)
```

Both forms are supported:

```python
@function_tool          # no parentheses
@function_tool()        # with parentheses — required when passing options

@function_tool(
    name_override="weather",
    description_override="Get the weather for a city.",
)
def get_weather(city: str) -> str: ...
```

**Supported argument types**: `str`, `int`, `float`, `bool`, `list`, `dict`, `Optional[T]`, `List[T]`, `Dict[K, V]`, `Annotated[T, Field(...)]`, Pydantic `BaseModel`, `dataclass`, `TypedDict`.

**Async tools** are fully supported:

```python
import asyncio

@function_tool
async def fetch_data(url: str) -> str:
    """Fetch content from a URL."""
    await asyncio.sleep(0)   # real async work here
    return f"Data from {url}"
```

**Rich argument descriptions** via `Annotated` or docstring:

```python
from typing import Annotated
from pydantic import Field

@function_tool
def search(
    query: Annotated[str, Field(description="What to search for")],
    max_results: Annotated[int, Field(description="Number of results")] = 5,
) -> str:
    """Search the web."""
    ...
```

---

### Tool base class

For full control, subclass `Tool` directly:

```python
from pydantic import BaseModel
from abzagent import Agent
from abzagent.core.tools import Tool

class SearchSchema(BaseModel):
    query: str
    max_results: int = 5

class SearchTool(Tool):
    name = "search"
    description = "Search the web for information."
    schema = SearchSchema

    def run(self, **kwargs) -> str:
        query = kwargs["query"]
        n = kwargs["max_results"]
        return f"Results for '{query}' (top {n}): ..."

agent = Agent(..., tools=[SearchTool()])
```

---

### FunctionTool (manual)

For OpenAI-compatible JSON-schema style tool definitions:

```python
from abzagent.core.tools import FunctionTool

tool = FunctionTool(
    name="calculator",
    description="Evaluate an arithmetic expression.",
    params_json_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. 2 + 3 * 4"}
        },
        "required": ["expression"],
    },
    python_fn=lambda expression: str(eval(expression)),  # replace with safe impl
)
```

Or with a JSON handler (receives raw args as a JSON string):

```python
import json

tool = FunctionTool(
    name="echo",
    description="Echo back the input.",
    on_invoke_tool=lambda ctx, args_json: json.loads(args_json)["text"],
)
```

---

### Built-in Tools

The SDK ships two ready-to-use tools in `abzagent.Tools`.

#### MathTool — safe arithmetic calculator

Evaluates arithmetic expressions using a stdlib `ast`-based whitelist evaluator.
Accepts `+ - * / ** % //` and parentheses. Rejects all other Python — imports,
function calls, attribute access, lambdas, etc.

```python
from abzagent.Tools.tools_math import MathTool

agent = Agent(
    name="MathBot",
    instructions="You solve arithmetic problems.",
    model="gemini-2.0-flash",
    tools=[MathTool()],
)

result = agent.run("What is (1.5 + 2.5) * 2?")
print(result.content)   # 8.0
```

Direct use:

```python
from abzagent.Tools.tools_math import safe_eval

print(safe_eval("2 + 3 * 4"))   # 14
print(safe_eval("(1.5 + 2.5) * 2"))   # 8.0
```

#### TimeTool — current time with timezone

```python
from abzagent.Tools.tools_time import TimeTool

agent = Agent(
    name="ClockBot",
    instructions="You tell the time.",
    model="gemini-2.0-flash",
    tools=[TimeTool()],
)

result = agent.run("What time is it in Tokyo?")
print(result.content)
```

Direct use:

```python
from abzagent.Tools.tools_time import TimeTool

tool = TimeTool()
print(tool.run(timezone="America/New_York"))   # 2026-07-29T10:23:45-04:00
print(tool.run())                              # UTC
```

Accepts any [IANA timezone](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) string (`"Europe/London"`, `"Asia/Tokyo"`, `"US/Pacific"`, etc.).

---

## Multi-Step / Iterative Mode

By default, `max_iterations=1` — one model call, one optional tool call, done.
Set `max_iterations` higher to let the agent loop: call tools, get results, keep
reasoning until it produces a final answer.

```python
agent = Agent(
    name="Researcher",
    instructions="Answer questions thoroughly. Use tools as needed.",
    model="gemini-2.0-flash",
    tools=[search_tool, calculator_tool],
    max_iterations=5,
)

result = agent.run("What is the GDP of Turkey divided by its population?")
print(result.content)
print(result.steps)   # see every model output and tool result
```

In each iteration the agent either:
- Emits a JSON tool call → SDK executes the tool → result fed back as next prompt
- Emits a plain text answer → loop ends and `AgentResult` is returned

If the iteration limit is reached without a final answer, the last model output is returned with a prefix note.

---

## Dynamic Instructions

`instructions` can be a function instead of a string. It is called at the start of every `run()` and receives a `RunContextWrapper` and the `Agent` itself, letting you build instructions dynamically from context, user data, or runtime state.

```python
from abzagent.core.agent import RunContextWrapper

def my_instructions(ctx: RunContextWrapper, agent) -> str:
    user_name = ctx.context.get("user_name", "there") if ctx.context else "there"
    return f"You are a helpful assistant. The user's name is {user_name}."

agent = Agent(
    name="PersonalBot",
    instructions=my_instructions,
    model="gemini-2.0-flash",
)

result = agent.run("Hello!", context={"user_name": "Abu"})
print(result.content)
```

Async instruction functions are supported:

```python
async def async_instructions(ctx, agent) -> str:
    return "Dynamic instructions from async source."
```

`RunContextWrapper` attributes:

| Attribute | Description |
|---|---|
| `.current_agent` | The agent running this turn |
| `.target_agent` | Same as current (used in handoff chains) |
| `.memory` | The agent's `Memory` instance |
| `.steps` | Steps accumulated so far in this `run()` call |
| `.context` | The `context` value passed to `agent.run()` |

---

## Structured Output

Set `output_type` to any Pydantic model to have the agent's response automatically parsed into that type. The parsed object is available on `AgentResult.parsed`.

```python
from pydantic import BaseModel
from abzagent import Agent

class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    condition: str

agent = Agent(
    name="WeatherParser",
    instructions="Extract weather info as JSON.",
    model="gemini-2.0-flash",
    output_type=WeatherReport,
)

result = agent.run("Istanbul: 24°C, sunny.")
report: WeatherReport = result.parsed
print(report.city)           # Istanbul
print(report.temperature_c)  # 24.0
```

---

## Guardrails

Guardrails are validation functions that run on input before the agent processes it and/or on output before it is returned. If a guardrail's tripwire is triggered, an exception is raised immediately.

### Input guardrail

```python
from abzagent.core.guardrails import input_guardrail, GuardrailFunctionOutput

@input_guardrail
def no_profanity(ctx, agent, user_input: str) -> GuardrailFunctionOutput:
    bad_words = ["spam", "hack"]
    triggered = any(w in user_input.lower() for w in bad_words)
    return GuardrailFunctionOutput(
        output_info={"checked": True},
        tripwire_triggered=triggered,
        reason="Profanity detected." if triggered else None,
    )

agent = Agent(
    name="SafeBot",
    instructions="Be helpful.",
    model="gemini-2.0-flash",
    input_guardrails=[no_profanity],
)

try:
    agent.run("How do I hack a server?")
except Exception as e:
    print(e)   # Input guardrail 'no_profanity' tripwire triggered.
```

### Output guardrail

```python
from abzagent.core.guardrails import output_guardrail, GuardrailFunctionOutput

@output_guardrail
def length_check(ctx, agent, output) -> GuardrailFunctionOutput:
    too_long = len(str(output)) > 5000
    return GuardrailFunctionOutput(
        output_info={"length": len(str(output))},
        tripwire_triggered=too_long,
        reason="Response too long." if too_long else None,
    )

agent = Agent(
    name="ConciseBot",
    instructions="Be brief.",
    model="gemini-2.0-flash",
    output_guardrails=[length_check],
)
```

Both sync and async guardrail functions are supported. A guardrail must return a `GuardrailFunctionOutput` — anything else raises `TypeError`.

---

## Handoffs

Handoffs let one agent transfer a conversation to another specialized agent. Each handoff is automatically registered as a `handoff_to_<agent_name>` tool on the host agent.

```python
from abzagent import Agent
from abzagent.core.handoffs import handoff

billing_agent = Agent(
    name="billing",
    instructions="You handle billing and payment questions.",
    model="gemini-2.0-flash",
)

support_agent = Agent(
    name="support",
    instructions="You handle general support. Transfer billing questions to billing.",
    model="gemini-2.0-flash",
    handoffs=[handoff(billing_agent)],
)

result = support_agent.run("I need help with my invoice.")
print(result.content)
```

### Handoff context filters

Clean up conversation history before it is passed to the receiving agent:

```python
from abzagent.extensions.handoffs_filter import remove_all_tools, keep_last_n_turns

# Remove tool messages from history
filtered = remove_all_tools(handoff_data)

# Keep only last 3 user/assistant turns
trimmed = keep_last_n_turns(3)(handoff_data)
```

### Handoff prompt helper

Add the recommended handoff instructions prefix to any agent's instructions:

```python
from abzagent.extensions.handoff_prompt import prompt_with_handoff_instructions

instructions = prompt_with_handoff_instructions(
    "You are a triage agent. Route to the right specialist."
)

agent = Agent(name="Triage", instructions=instructions, ...)
```

---

## Agent as a Tool

Any agent can be wrapped as a tool and used inside another agent, enabling nested / hierarchical agent architectures.

```python
from abzagent import Agent

sub_agent = Agent(
    name="Summarizer",
    instructions="You summarize long text into 3 bullet points.",
    model="gemini-2.0-flash",
)

# Wrap it as a tool
summarizer_tool = sub_agent.as_tool(
    tool_name="summarize",
    tool_description="Summarize a long piece of text into bullet points.",
)

orchestrator = Agent(
    name="Orchestrator",
    instructions="You coordinate tasks. Use the summarize tool when needed.",
    model="gemini-2.0-flash",
    tools=[summarizer_tool],
)

result = orchestrator.run("Summarize the history of the internet for me.")
print(result.content)
```

---

## Verbose Mode

Set `verbose=True` to print tool execution details to stdout as the agent runs. Useful for debugging multi-step agents.

```python
agent = Agent(
    name="DebugAgent",
    instructions="Be helpful.",
    model="gemini-2.0-flash",
    tools=[my_tool],
    verbose=True,
    max_iterations=4,
)
```

Output includes the tool name and kwargs for every tool call.

---

## CLI

The SDK installs an `abz-agents` command.

### `abz-agents setup`

Interactive wizard that scaffolds a new project:

```
$ abz-agents setup

Welcome To ABZ Agent SDK — Project setup

Select Model Provider [gemini/groq] (default: gemini): groq
Enter your GROQ_API_KEY (leave blank to fill later): gsk_...
Agent name [My Agent]: SupportBot
Agent instructions [Be helpful and concise.]: You handle customer support.
Model id [qwen/qwen3-32b]:
Starter file name [agent.py]:

✓ Wrote .env
✓ Created agent.py

Next steps:
  1) Ensure your .env has GROQ_API_KEY set
  2) Run:  abz-agents run agent.py
```

### `abz-agents run <file.py>`

Runs any Python file as `__main__`:

```bash
abz-agents run agent.py
abz-agents run examples/quickstart.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Required when using Gemini models |
| `GROQ_API_KEY` | Required when using Groq models |
| `ABZ_MODEL` | Default model override (default: `models/gemini-1.5-pro`) |
| `ABZ_TEMPERATURE` | Sampling temperature (default: `0.4`) |
| `ABZ_MAX_ITERS` | Default max iterations (default: `4`) |
| `ABZ_VERBOSE` | Enable verbose mode by default (`1` = on, `0` = off) |

All variables are loaded from `.env` automatically.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history including the v0.3.1 security fix.