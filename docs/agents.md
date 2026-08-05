# Agents

The `Agent` class is the core orchestrator of ABZ Agent SDK. An agent bundles a name, a
system prompt (`instructions`), an LLM provider/model, an optional set of `tools`, and an
optional `Memory` buffer, and exposes a single entrypoint — `run()` — that drives one
question/answer (or multi-step tool-using) cycle.

> **How it actually works under the hood:** ABZ Agent SDK does **not** use the native
> function-calling APIs of Gemini or Groq. Instead, `Agent` builds one flat text prompt
> (system instructions + a manifest of available tools + conversation history + the user
> message), sends it to the provider's plain text-generation endpoint, and then scans the
> model's raw text output for a single-line JSON object of the shape
> `{"tool": "<name>", "args": {...}}`. If one is found, the SDK executes the matching tool
> and (in multi-step mode) feeds the result back in as the next turn. This is a convention
> enforced entirely through the system prompt, not a provider-level tool-calling contract.
> Keep this in mind when debugging: if a model ever prints extra text around the JSON blob,
> or wraps it in markdown fences, tool-call detection can fail.

## Import

```python
from abzagent import Agent, AgentResult
```

## Quick start

```python
from abzagent import Agent

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gemini-2.0-flash",
)

result = agent.run("What is the capital of France?")
print(result.content)
```

## Constructor

```python
Agent(
    *,
    name: str,
    instructions: Union[str, InstructionsFn],
    model: Optional[str] = "gemini-2.0-flash",
    tools: Optional[List[Union[Tool, Callable]]] = None,
    handoffs: Optional[List[Union[Agent, Handoff]]] = None,
    memory: Optional[Memory] = None,
    verbose: bool = False,
    max_iterations: Optional[int] = None,
    api_key: Optional[str] = None,
    validate_model: bool = False,
    include_experimental: bool = True,
    output_type: Optional[Type[Any]] = None,
    input_guardrails: Optional[Sequence[Any]] = None,
    output_guardrails: Optional[Sequence[Any]] = None,
)
```

All parameters are keyword-only.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | — | **Required.** Raises `ValueError` if empty. Used in the prompt (`[AGENT NAME]:`) and as the display name in handoff tool names. |
| `instructions` | `str` \| `InstructionsFn` | — | **Required.** Raises `ValueError` if empty/blank string. The agent's system prompt, either static or computed per-call (see [Dynamic instructions](#dynamic-instructions)). |
| `model` | `str \| None` | `"gemini-2.0-flash"` | Any Gemini or Groq model id. Falsy values or the literal string `"auto"` resolve to `"gemini-2.0-flash"`. Anything else is used as-is — there is currently no validation that the model id is real (see [Model resolution caveats](#model-resolution-caveats)). |
| `tools` | `list[Tool \| Callable]` | `None` | Mix of `Tool` instances and plain Python functions. Plain functions are auto-wrapped with `function_tool()`. See [Tools](tools.md). |
| `handoffs` | `list[Agent \| Handoff]` | `None` | ⚠️ **Not currently functional** — see [Handoffs](#handoffs-not-currently-functional) below. |
| `memory` | `Memory \| None` | `None` | If omitted, a fresh `Memory()` is created automatically. See [Memory](memory.md). |
| `verbose` | `bool` | `False` | When `True`, prints the tool name and resolved kwargs to stdout every time a tool executes (`_execute_tool`). |
| `max_iterations` | `int \| None` | `None` → `1` | `1` (or unset) means **single-turn mode**: one model call, at most one tool call, then return. `>1` enables the iterative tool-use loop. See [Single-turn vs. multi-step mode](#single-turn-vs-multi-step-mode). |
| `api_key` | `str \| None` | `None` | Explicit API key, overrides environment variables. |
| `validate_model` | `bool` | `False` | Accepted for forward-compatibility but **currently has no effect** — the internal model resolver ignores it. |
| `include_experimental` | `bool` | `True` | Same as above — accepted but currently unused. |
| `output_type` | `Type \| None` | `None` | Enables structured output. When set, `AgentResult.parsed` is a validated instance of this type. See [Structured output](#structured-output). |
| `input_guardrails` | `Sequence` | `None` | List of `@input_guardrail`-decorated functions, run before the model sees the user message. **Fully functional.** |
| `output_guardrails` | `Sequence` | `None` | List of `@output_guardrail`-decorated functions. ⚠️ **Accepted and stored but never executed** — see [Guardrails](#guardrails). |

### API key resolution order

1. The `api_key` constructor argument, if given.
2. Otherwise, an environment variable — `GROQ_API_KEY` if the resolved provider is Groq,
   `GEMINI_API_KEY` if Gemini. `.env` is loaded automatically via `python-dotenv` at import
   time, so a `.env` file in the working directory is picked up with no extra setup.
3. If still empty, the constructor raises `RuntimeError(f"{key_name} missing — set in env or .env")`.

### Provider auto-detection

`Agent` never takes a `provider` argument — it infers Gemini vs. Groq purely from the
`model` string, via `SDKConfig.detect_provider`:

```python
groq_patterns = ["qwen/", "llama", "mixtral", "deepseek", "gemma2", "gemma-"]
```

Any model name containing one of those substrings routes to `GroqProvider`; everything
else routes to `GeminiProvider`.

```python
Agent(..., model="gemini-2.0-flash")          # -> Gemini
Agent(..., model="qwen/qwen3-32b")             # -> Groq
Agent(..., model="llama-3.3-70b-versatile")    # -> Groq
```

### Model resolution caveats

Internally, `_resolve_model_param` does nothing more than:

```python
if not model or str(model).strip().lower() == "auto":
    return "gemini-2.0-flash"
return str(model)
```

`validate_model` and `include_experimental` are accepted for API compatibility but are not
read by this function — passing an invalid or unsupported model id will not fail fast at
construction time; it will only surface as a provider-level error once `run()` is called.

## Public methods

### `agent.run(user_message: str, *, context: Any = None) -> AgentResult`

The main entrypoint. One call to `run()`:

1. Wraps the call in a `RunContextWrapper(current_agent=self, target_agent=self, memory=self.memory, steps=[], context=context)`.
2. Executes any `input_guardrails` (raises `InputGuardrailTripwireTriggered` if one trips).
3. Records the user message into `Memory` via `memory.remember("user", user_message)`.
4. Runs either single-turn or iterative mode (below).

`context` is an arbitrary value forwarded to dynamic instruction functions and guardrails
through `RunContextWrapper.context` — it is not interpreted by the SDK itself.

### `agent.register_tool(tool: Tool | Callable) -> None`

Registers an additional tool after construction. Accepts the same inputs as the `tools=`
constructor argument (a `Tool` instance or a plain callable, auto-wrapped). If a tool with
the same name already exists, it is silently overwritten (`self.tools[t.name] = t`).

```python
agent.register_tool(my_tool)
```

### `agent.as_tool(*, tool_name: str, tool_description: str) -> Tool`

Wraps the entire agent as a `Tool` so it can be handed to another agent's `tools=[...]`,
enabling nested/hierarchical agent architectures. The generated tool takes a single
`message: str` argument and, when invoked, calls `outer_agent.run(message).content`.

```python
sub_agent = Agent(name="Summarizer", instructions="Summarize text in 3 bullets.")

summarizer_tool = sub_agent.as_tool(
    tool_name="summarize",
    tool_description="Summarize a long piece of text into bullet points.",
)

orchestrator = Agent(
    name="Orchestrator",
    instructions="Coordinate tasks. Use the summarize tool when needed.",
    tools=[summarizer_tool],
)
```

## `AgentResult`

```python
class AgentResult:
    content: str        # final text answer (raw JSON text when output_type is set)
    steps: list[str]    # every raw model output / tool-result string, in call order
    parsed: Any          # instance of output_type, or None if output_type wasn't set
```

```python
result = agent.run("What is 2 + 2?")
print(result.content)   # "4"
print(result.steps)     # ["4"]  (or, if a tool ran, [model_output, tool_result])
```

## Single-turn vs. multi-step mode

Controlled by `max_iterations` (default `1`):

**Single-turn (`max_iterations <= 1`, the default)**

1. Builds one prompt, calls the model once.
2. If the model's output is a tool-call JSON blob, the tool is executed and its result
   **becomes `AgentResult.content` directly** — there is no follow-up model call to turn the
   tool result into a natural-language answer. `steps` will contain `[model_output, tool_result]`.
3. If the model's output is plain text, that text is `AgentResult.content` and `steps` contains just `[model_output]`.

**Iterative (`max_iterations > 1`)**

Loops up to `max_iterations` times. On each pass:
- The first iteration's prompt uses the real user message; every subsequent iteration's
  prompt uses the literal string `"Continue."` plus a `TOOL RESULT (<tool>): <result>` line
  appended to memory.
- If the model emits a tool-call JSON blob, the tool runs and the loop continues.
- If the model emits plain text, the loop ends and that text is the final `AgentResult.content`.
- If `max_iterations` is exhausted without the model producing a plain-text answer, the
  result is the last step, prefixed with `"Reached iteration limit without final answer.\n\n"`.

```python
agent = Agent(
    name="Researcher",
    instructions="Answer thoroughly. Use tools as needed.",
    tools=[search_tool, calculator_tool],
    max_iterations=5,
)

result = agent.run("What is the GDP of Turkey divided by its population?")
print(result.content)
print(result.steps)   # every model output and tool result along the way
```

## Dynamic instructions

`instructions` may be a function instead of a plain string. It is called once per `run()`
call (not once per constructor call), receiving a `RunContextWrapper` and the `Agent`
instance, and must return a non-empty string. Both sync and async functions are supported.

```python
from abzagent.core.agent import RunContextWrapper

def my_instructions(ctx: RunContextWrapper, agent) -> str:
    user_name = ctx.context.get("user_name", "there") if ctx.context else "there"
    return f"You are a helpful assistant. The user's name is {user_name}."

agent = Agent(name="PersonalBot", instructions=my_instructions)

result = agent.run("Hello!", context={"user_name": "Abu"})
```

```python
async def async_instructions(ctx, agent) -> str:
    return "Dynamic instructions from an async source."
```

Async instruction functions are executed with `asyncio.run`; if already inside a running
event loop (e.g. Jupyter), the SDK falls back to `nest_asyncio`.

### `RunContextWrapper`

Passed to dynamic instruction functions and guardrails.

| Attribute | Type | Description |
|---|---|---|
| `.current_agent` | `Agent` | The agent running this turn. |
| `.target_agent` | `Agent` | Same object as `.current_agent` today (reserved for future handoff-chain use). |
| `.memory` | `Memory` | The agent's `Memory` instance. |
| `.steps` | `list[str]` | Steps accumulated so far in this `run()` call. |
| `.context` | `Any` | The `context` value passed to `agent.run(..., context=...)`. |

## Structured output

Set `output_type` to get a validated Python object back instead of raw text — no manual
prompting for JSON, no manual parsing.

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Extractor",
    instructions="Extract the calendar event described by the user.",
    model="gemini-2.0-flash",
    output_type=CalendarEvent,
)

result = agent.run("Standup on 2026-08-05 with Abu and Sara.")
result.parsed.name          # "Standup"
result.parsed.participants  # ["Abu", "Sara"]
```

`output_type` accepts anything `pydantic.TypeAdapter` accepts: a `BaseModel` subclass
(the common case), a `dataclass`, a `TypedDict`, or a plain type like `list[str]` or `int`.
Types that aren't naturally a JSON object (e.g. `list[str]`) are transparently wrapped in a
single-key JSON envelope on the wire — you never see the envelope; `result.parsed` is always
a plain instance of `output_type`.

### How it works

1. **Schema generation** — a JSON Schema is generated once from `output_type` (via
   `pydantic.TypeAdapter`) and cached on the agent as an `AgentOutputSchema`
   (`abzagent.core.output.AgentOutputSchema`).
2. **Prompting** — the schema is appended to the system prompt automatically. You never
   write "respond in JSON" yourself.
3. **Generation** — each provider is asked to produce JSON using whatever native
   structured-output capability it has (see below).
4. **Validation** — the raw text is parsed and validated against `output_type`. On success,
   `AgentResult.parsed` is the validated object and `AgentResult.content` is the raw JSON text.
5. **Repair retry** — if validation fails (bad JSON, missing/wrong-typed fields), the SDK
   automatically sends the model its previous output plus the validation error and asks for a
   corrected response, once, with no code on your end. This handles the vast majority of
   real-world hiccups by itself.

You don't need to write any error handling for this feature to work — steps 1–5 all happen
for you. The only thing worth knowing: on the rare occasion a model still can't produce valid
output after the automatic retry, `run()` raises `abzagent.ModelBehaviorError` instead of
silently handing you broken data. It's a normal Python exception — catch it if you want a
custom message, otherwise ignore it like any other error:

```python
from abzagent import ModelBehaviorError

try:
    result = agent.run("...")
except ModelBehaviorError as e:
    print(f"Model failed to produce valid {CalendarEvent.__name__}: {e}")
```

### Provider-specific behavior

Both providers reach the same result through different native mechanisms — this is internal;
the public API (`output_type` in, `result.parsed` out) is identical either way:

- **Gemini** — uses `response_mime_type="application/json"` plus `response_json_schema` (the
  exact generated schema) via `google-genai`'s native structured-output support. When the
  agent has no tools, this is schema-locked: the model is constrained to emit exactly the
  declared shape.
- **Groq** — uses `response_format={"type": "json_object"}` (broadly supported across Groq
  models). Groq's stricter `json_schema` response format exists but is only available on
  select models, so the SDK doesn't depend on it; the same schema-in-prompt + Pydantic
  validation + repair-retry pipeline above does the exact-shape enforcement instead.

### Structured output + tools

`output_type` and `tools` can be used together. While tools are active, the SDK does not
lock generation to the exact output schema on every turn (a tool-call JSON blob needs to
remain a valid response too) — schema validation only applies once the model produces its
final, non-tool-call answer. One caveat inherited from the SDK's [prompt-based tool-calling
convention](#agents): in **single-turn mode** (`max_iterations <= 1`), if the model's first
response is a tool call, `AgentResult.content` becomes that tool's raw output directly (see
[Single-turn vs. multi-step mode](#single-turn-vs-multi-step-mode)) and `result.parsed` stays
`None` for that call — structured validation never runs on a raw tool result. Use
`max_iterations > 1` if you want the agent to reliably follow up a tool call with a
schema-validated final answer.

## Guardrails

Guardrails are validation functions wrapping the input and/or output of a `run()` call.

```python
from abzagent.core.guardrails import input_guardrail, GuardrailFunctionOutput

@input_guardrail
def no_profanity(ctx, agent, user_input: str) -> GuardrailFunctionOutput:
    triggered = "hack" in user_input.lower()
    return GuardrailFunctionOutput(
        output_info={"checked": True},
        tripwire_triggered=triggered,
        reason="Profanity detected." if triggered else None,
    )

agent = Agent(
    name="SafeBot",
    instructions="Be helpful.",
    input_guardrails=[no_profanity],
)

try:
    agent.run("How do I hack a server?")
except Exception as e:
    print(e)   # Input guardrail 'no_profanity' tripwire triggered.
```

- A guardrail function must return `GuardrailFunctionOutput(output_info, tripwire_triggered, reason=None)`; returning anything else raises `TypeError`.
- If `tripwire_triggered` is `True`, `run()` raises `InputGuardrailTripwireTriggered` (input) — this happens **before** the user message is sent to the model.
- Both sync and async guardrail functions are supported.

**⚠️ Output guardrails are accepted by the constructor and stored on `self.output_guardrails`,
but `Agent.run()` never calls `run_output_guardrails()` on them.** Passing
`output_guardrails=[...]` today has no observable effect — no exception will ever be raised
from an output guardrail, and `length_check`-style examples will silently do nothing. Treat
output guardrails as not-yet-wired-up rather than enforced.

## Handoffs (not currently functional)

The SDK's public surface includes a `handoffs=` constructor argument and a
`abzagent.core.handoffs.handoff()` factory intended to let one agent transfer a
conversation to another specialized agent (each registered automatically as a
`handoff_to_<agent_name>` tool). **In the current codebase this feature is broken**:
`core/handoffs.py` imports from `core/abz_handoff_core.py`, a module that does not exist
anywhere in the package. `Agent`'s import of the handoffs module is wrapped in a
`try/except Exception`, so the failure is silently swallowed at import time and replaced
with a stub:

```python
def handoff(agent):
    raise RuntimeError("Handoffs not available; module missing.")
```

Any code that calls `handoff(some_agent)` — including the pattern shown in older examples —
will raise `RuntimeError("Handoffs not available; module missing.")` immediately. Do not
document or demo the `handoffs=` parameter as working until `abz_handoff_core.py` is
restored (and `HandoffInputData`, referenced by
`abzagent/extensions/handoffs_filter.py`, is added to `handoffs.py` — that file also
fails to import today for the same reason).

## Verbose mode

```python
agent = Agent(
    name="DebugAgent",
    instructions="Be helpful.",
    tools=[my_tool],
    verbose=True,
    max_iterations=4,
)
```

When `verbose=True`, every tool execution prints `Executing tool <name> with kwargs=<kwargs>` to stdout — useful when debugging multi-step tool-using agents.

## Provider error-handling asymmetry

Worth knowing when writing error-handling code around `agent.run()`:

- `GeminiProvider.generate()` lets exceptions from the underlying `google-genai` client propagate normally.
- `GroqProvider.generate()` catches **all** exceptions internally and returns the string `"[Groq Error] {str(e)}"` as if it were a valid model response, rather than raising.

This means a Groq-backed agent can "succeed" (no exception) with an API/network error baked
directly into `result.content` or `result.steps`, while a Gemini-backed agent under the same
failure would raise. If you need uniform error handling across both providers, check
`result.content` for a `"[Groq Error]"` prefix in addition to wrapping `run()` in `try/except`.

## Known limitations summary

| Feature | Status |
|---|---|
| `output_type` / `result.parsed` | **Implemented** — see [Structured output](#structured-output) |
| `output_guardrails` | Accepted, stored, never invoked |
| `handoffs=` / `handoff()` | Raises `RuntimeError` — missing internal module |
| `validate_model`, `include_experimental` | Accepted, currently no-ops |
| Native provider function-calling | Not used — tool calls are parsed from prompt text via JSON convention |
