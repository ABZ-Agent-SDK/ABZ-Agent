# Agents

The `Agent` class is the core orchestrator of ABZ Agent SDK. An agent bundles a name, a
system prompt (`instructions`), an LLM provider/model, an optional set of `tools`, and an
optional `Memory` buffer, and exposes a single entrypoint — `run()` — that drives one
question/answer (or multi-step tool-using) cycle.

> **How it actually works under the hood:** `Agent` builds one flat text prompt (system
> instructions + conversation history + the user message) each turn and calls
> `provider.generate()`. Tool calls (including [Handoffs](#handoffs), which are just tools)
> are detected using each provider's **native function/tool-calling API** when the provider
> supports it — currently both Gemini and Groq do (`ModelProvider.supports_native_tools`,
> `True` for both). Each registered `Tool`'s schema (via `Tool.schema.model_json_schema()`)
> is translated into the provider's own tool-definition format and sent structurally
> alongside the prompt; the tool call comes back as a real field on the response
> (`GenerationResult.tool_calls`), not text to parse. This is far more reliable than text
> parsing — the provider enforces the shape, not a prompt convention.
>
> A provider that genuinely lacks tool-calling support (`supports_native_tools = False`)
> falls back to the older convention: a text manifest of tools is added to the system
> prompt, and the model's raw output is scanned for a single-line JSON object shaped
> `{"tool": "<name>", "args": {...}}`. This fallback path still exists and is fully
> functional, but with both real providers supporting native calling, it's not the path
> you'll hit in practice. Tool *results* are still fed back as plain text
> (`TOOL RESULT (<tool>): <result>`) in both modes — this SDK doesn't (yet) model a full
> native multi-turn tool-conversation transcript.

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
    tool_input_guardrails: Optional[Sequence[Any]] = None,
    tool_output_guardrails: Optional[Sequence[Any]] = None,
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
| `input_guardrails` | `Sequence` | `None` | List of `@input_guardrail`-decorated functions, run before the model sees the user message. |
| `output_guardrails` | `Sequence` | `None` | List of `@output_guardrail`-decorated functions, run on the final answer before `run()` returns. See [Guardrails](#guardrails). |
| `tool_input_guardrails` | `Sequence` | `None` | List of `@tool_input_guardrail`-decorated functions, run before each tool call executes. See [Tool guardrails](#tool-guardrails). |
| `tool_output_guardrails` | `Sequence` | `None` | List of `@tool_output_guardrail`-decorated functions, run after each tool call returns. See [Tool guardrails](#tool-guardrails). |

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

### `agent.run(user_message=None, *, context=None, interactive=False) -> AgentResult | None`

The SDK's single execution entrypoint — for a single request, one call to `run(user_message)`:

1. Wraps the call in a `RunContextWrapper(current_agent=self, target_agent=self, memory=self.memory, steps=[], context=context)`.
2. Executes any `input_guardrails` (raises `InputGuardrailTripwireTriggered` if one trips).
3. Records the user message into `Memory` via `memory.remember("user", user_message)`.
4. Runs either single-turn or iterative mode (below).

`context` is an arbitrary value forwarded to dynamic instruction functions and guardrails
through `RunContextWrapper.context` — it is not interpreted by the SDK itself.

`run()` also supports an interactive terminal mode via `interactive=True`, covered right
after `as_tool()` below.

### `agent.register_tool(tool: Tool | Callable) -> None`

Registers an additional tool after construction. Accepts the same inputs as the `tools=`
constructor argument (a `Tool` instance or a plain callable, auto-wrapped). If a tool with
the same name already exists, it is silently overwritten (`self.tools[t.name] = t`).

```python
agent.register_tool(my_tool)
```

### Interactive mode: `agent.run(interactive=True)`

No manual `while True` / `input()` loop needed:

```python
agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
agent.run(interactive=True)
```

`interactive=True` starts a REPL that calls `run(user_input)` — this same method, in
single-request mode — for every message; it returns `None` when the session ends rather
than an `AgentResult`. There's no separate inference path: every feature that already
works with `run("...")` (memory, tools, structured output, handoffs) works automatically
in interactive mode too, because each turn just is a normal `run()` call.

- Typing `exit` or `quit` (case-insensitive) ends the session with a goodbye message.
- Ctrl+C or Ctrl+D/EOF at any point — including mid-request — exits cleanly instead of printing a traceback.
- An exception from a turn's `run()` call (e.g. a transient provider error) is caught, printed
  as `[Error] ...`, and the loop continues rather than crashing the whole session.
- Blank input is silently skipped (re-prompts without calling `run()`).
- `context` is forwarded to every per-turn `run()` call exactly as it would be if you called `run()` yourself.
- Calling `run()` with neither a `user_message` nor `interactive=True`, or with both at once, raises `ValueError`.
- Each response line is printed as `{responder}: {content}`, where `{responder}` is
  `result.last_agent.name` — the agent that actually produced the answer, not necessarily the
  agent `interactive=True` was called on. If a turn triggers a handoff, the line after the
  `🔄 Handoff` diagram is labeled with the specialist's name, not the router's.

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
    content: str               # final text answer (raw JSON text when output_type is set)
    steps: list[str]           # every raw model output / tool-result string, in call order
    parsed: Any                 # instance of output_type, or None if output_type wasn't set
    last_agent: Agent | None    # whichever agent actually produced `content` (see Handoffs)
```

`last_agent` is `self` unless a [handoff](#handoffs) occurred, in which case it's whichever
agent in the chain actually produced the final answer.

```python
result = agent.run("What is 2 + 2?")
print(result.content)   # "4"
print(result.steps)     # ["4"]  (or, if a tool ran, [model_output, tool_result])
```

## Single-turn vs. multi-step mode

Controlled by `max_iterations` (default `1`):

**Single-turn (`max_iterations <= 1`, the default)**

1. Builds one prompt, calls the model once.
2. If the model calls a tool (via native tool-calling, or the JSON-blob fallback
   convention — see [Tools](tools.md)), the tool is executed and its result **becomes
   `AgentResult.content` directly** — there is no follow-up model call to turn the tool
   result into a natural-language answer. `steps` will contain `[tool_call_repr, tool_result]`.
3. If the model's output is plain text, that text is `AgentResult.content` and `steps` contains just `[model_output]`.

**Iterative (`max_iterations > 1`)**

Loops up to `max_iterations` times. On each pass:
- The first iteration's prompt uses the real user message; every subsequent iteration's
  prompt uses the literal string `"Continue."` plus a `TOOL RESULT (<tool>): <result>` line
  appended to memory.
- If the model calls a tool, it runs and the loop continues.
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
lock generation to the exact output schema on every turn — schema-locking and native tool
calling are never requested in the same provider call (see `strict` in the callout at the
top of this page) — schema validation only applies once the model produces its final,
non-tool-call answer. One caveat inherited from the SDK's [tool-calling
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

- A guardrail function must return `GuardrailFunctionOutput(tripwire_triggered, output_info=None, reason=None)`; returning anything else raises `TypeError`. `output_info` defaults to `None`, so `GuardrailFunctionOutput(tripwire_triggered=False)` alone is valid — you don't need to pass `output_info` when there's nothing to report.
- If `tripwire_triggered` is `True`, `run()` raises `InputGuardrailTripwireTriggered` (input) — this happens **before** the user message is sent to the model.
- Both sync and async guardrail functions are supported.
- The `agent` parameter may be omitted from any guardrail function's signature if
  you don't need it — `def my_guardrail(ctx, user_input): ...` works exactly the
  same as the full `def my_guardrail(ctx, agent, user_input): ...`. This applies
  to all four guardrail kinds (input, output, tool input, tool output).

### Output guardrails

```python
from abzagent.core.guardrails import output_guardrail, GuardrailFunctionOutput

@output_guardrail
def length_check(ctx, agent, output) -> GuardrailFunctionOutput:
    triggered = len(str(output)) > 500
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=triggered,
        reason="Response too long." if triggered else None,
    )

agent = Agent(
    name="Terse",
    instructions="Be helpful.",
    output_guardrails=[length_check],
)

try:
    agent.run("Explain the whole history of computing.")
except Exception as e:
    print(e)   # Output guardrail 'length_check' tripwire triggered.
```

Output guardrails run on the agent's final answer, immediately before `run()` returns —
whether that answer comes back after a single turn, after the iterative tool-use loop
finishes, or from the "reached iteration limit" fallback path. A trip raises
`OutputGuardrailTripwireTriggered`. If `output_type` is set, the guardrail receives the
already-validated, parsed object (`AgentResult.parsed`) instead of raw text — the same value
your own code would see on `result.parsed`.

## Tool guardrails

Tool guardrails wrap a single tool call, agent-level (not per-tool): `tool_input_guardrails=`
runs before the tool executes, `tool_output_guardrails=` runs after it returns.

```python
from abzagent.core.guardrails import tool_input_guardrail, tool_output_guardrail, GuardrailFunctionOutput

@tool_input_guardrail
def block_forbidden_item(ctx, agent, call, kwargs) -> GuardrailFunctionOutput:
    triggered = kwargs.get("item") == "forbidden"
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered=triggered,
        reason="That item can't be looked up." if triggered else None,
    )

agent = Agent(
    name="Shop",
    instructions="Help with prices.",
    tools=[lookup_price],
    tool_input_guardrails=[block_forbidden_item],
)
```

- `kwargs` are the tool's already-validated arguments (after `Tool.parse_args()`), not the
  model's raw args — a guardrail sees the same typed values the tool function itself would
  receive.
- Unlike input/output guardrails, a tripped tool guardrail **degrades gracefully by default —
  it does not raise.** `tool.run()` is skipped (input-side) or its real result is discarded
  (output-side), and the model instead receives `"[Tool Guardrail Blocked] {reason}"` as the
  tool's result, the same way an ordinary tool error is fed back today. This lets the model
  react and try something else instead of aborting the whole run.
- To hard-abort the run when a tool guardrail trips, `raise ToolGuardrailTripwireTriggered(...)`
  directly from inside the guardrail function — it propagates out of `run()` uncaught.
- A guardrail that only cares about one tool can check `call.tool` itself and no-op otherwise
  (return `tripwire_triggered=False`) — there's no per-tool guardrail config to reach for.
- Tool guardrails don't gate handoffs — a `transfer_to_...` tool call is intercepted before
  `_execute_tool()` is ever reached and is governed solely by the handoff machinery
  (`CircularHandoffError`, `MaxHandoffDepthExceededError`).
- `agent.as_tool()`-wrapped sub-agents are ordinary tool calls, so tool guardrails apply to
  them automatically.
- Both sync and async guardrail functions are supported, for all four guardrail decorators.

### Guardrail exceptions

```python
from abzagent.core.guardrails import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ToolGuardrailTripwireTriggered,
)
```

| Exception | When |
|---|---|
| `InputGuardrailTripwireTriggered` | An `input_guardrails` function tripped — raised before the user message reaches the model. |
| `OutputGuardrailTripwireTriggered` | An `output_guardrails` function tripped — raised right before `run()` returns the final answer. |
| `ToolGuardrailTripwireTriggered` | Not raised by the framework itself. Raise it yourself from inside a tool guardrail function if you want that trip to hard-abort the run instead of degrading gracefully (see above). |

All three subclass `RuntimeError`.

## Handoffs

Set `handoffs=[...]` to let an agent transfer a conversation to another, more specialized
agent — routing, memory, and context transfer are all automatic. The developer never
manually manages routing.

```python
from abzagent import Agent

research_agent = Agent(name="Research", instructions="Research topics thoroughly.")
writer_agent = Agent(name="Writer", instructions="Write clear, engaging copy.")
review_agent = Agent(name="Review", instructions="Review and polish drafts.")

planner = Agent(
    name="Planner",
    instructions="Route tasks to the correct specialist.",
    handoffs=[research_agent, writer_agent, review_agent],
)

result = planner.run("Write a short blog post about black holes.")
result.content       # the specialist's answer
result.last_agent    # whichever agent actually produced it (e.g. the Writer agent)
```

### How it works

1. Each entry in `handoffs=[...]` is registered as a `transfer_to_<agent_name>` tool on the
   host agent (bare `Agent` instances are wrapped automatically; see below for the
   `handoff()` factory for customization). This reuses the SDK's existing tool-calling
   mechanism exactly — handoffs are detected and dispatched exactly like any other tool
   call (native or fallback), no separate protocol.
2. When the model calls a `transfer_to_...` tool, `Agent.run()` recognizes it as a handoff
   (rather than a normal tool) and, instead of feeding a tool result back into its own
   conversation, replays its conversation history into the target agent's `Memory` and
   recursively calls the target agent's `run()`.
3. The target agent's own `AgentResult` — its `content`, `parsed`, tools, everything — is
   returned directly as the result of the *original* `run()` call. A target agent can
   itself have `handoffs=[...]` and hand off further; chains are supported.
4. `result.last_agent` always reflects whichever agent actually produced `result.content`,
   even through a multi-hop chain.

A router-focused prompt hint is automatically added to the system prompt whenever an agent
has `handoffs=[...]` configured — framed as "you are a router first," with an explicit rule
for when to transfer vs. answer directly, positioned as the last thing before the user's
message (models weight the end of the prompt more heavily). You don't need to write any of
this into `instructions` yourself.

Each transfer tool's description is also generated automatically from the *target* agent's
own `instructions` (truncated if long) — so `handoffs=[writer_agent]` where `writer_agent`
was constructed with `instructions="Write blogs and articles."` gives the routing model a
real "use this agent when..." hint with zero extra code. Use `handoff(target,
tool_description_override=...)` if you want a different, hand-written description instead.

When `agent.run(interactive=True)` is used, each handoff prints a small arrow diagram
(`🔄 Handoff\n\n{from}\n   │\n   ▼\n{to}`) as it happens — automatically, nothing to print
yourself. This never happens during a plain `agent.run("...")` call; the SDK never writes to
stdout outside an interactive session.

### The `handoff()` factory (customization)

A bare `Agent` in `handoffs=[...]` covers the common case. Use `handoff()` for more control:

```python
from abzagent import Agent, handoff
from pydantic import BaseModel

class EscalationData(BaseModel):
    reason: str

def on_handoff(ctx, data: EscalationData):
    print(f"Escalating: {data.reason}")

support_agent = Agent(
    name="Support",
    instructions="Handle general support.",
    handoffs=[
        handoff(
            billing_agent,
            tool_name_override="escalate_to_billing",   # default: transfer_to_<slug(name)>
            tool_description_override="Escalate billing issues.",
            input_type=EscalationData,   # LLM must supply these fields to trigger the handoff
            on_handoff=on_handoff,        # called right before the transfer happens
            input_filter=None,            # see "Context filters" below
        ),
    ],
)
```

- `input_type` (a Pydantic model) makes the LLM supply structured arguments when triggering
  the handoff — validated the same way regular tool arguments are (via `Tool.parse_args()`).
  If required fields are missing, the handoff does **not** happen; the model gets a
  graceful "I need the following information..." message instead, same as any other tool
  with bad arguments.
- `on_handoff(ctx, data)` (or `on_handoff(ctx)` if no `input_type`) runs right before control
  transfers, receiving a [`RunContextWrapper`](#runcontextwrapper) with `target_agent` set.
- Without `input_type`, the model may optionally supply a free-form `message` argument,
  which becomes the continuation prompt sent to the target agent.

### Context filters

By default, the target agent inherits the full prior conversation. Trim or transform it
with `input_filter`:

```python
from abzagent import handoff
from abzagent.extensions.handoffs_filter import remove_all_tools, keep_last_n_turns

handoff(billing_agent, input_filter=remove_all_tools)   # drop prior tool messages
handoff(billing_agent, input_filter=keep_last_n_turns(3))  # keep only the last 3 messages
```

An `input_filter` is any `HandoffInputData -> HandoffInputData` function
(`abzagent.core.handoffs.HandoffInputData` wraps a `messages: list[Message]`).

If the source and target agent already share the same `Memory` instance (`Agent(memory=shared)`
on both), history is never replayed a second time — it's already there.

One caveat worth knowing: the SDK never writes an agent's own final reply into `Memory`
today (only `user` and `tool` roles are ever recorded) — this is a pre-existing behavior,
unrelated to handoffs. A handed-off agent sees what the user said and what tools returned,
not what a prior agent said back.

### Errors

```python
from abzagent import CircularHandoffError, MaxHandoffDepthExceededError, InvalidHandoffTargetError
```

| Exception | When |
|---|---|
| `InvalidHandoffTargetError` | Raised immediately at `handoff(...)`/`Agent(handoffs=[...])` construction time if a target isn't an `Agent`. |
| `CircularHandoffError` | An agent tried to hand off to an agent already in the current chain (including itself). |
| `MaxHandoffDepthExceededError` | A handoff chain exceeded `abzagent.core.handoffs.MAX_HANDOFF_DEPTH` (default `5`). |

All three subclass `RuntimeError` — the same category as guardrail tripwires (a structural
safety trip that aborts the run), not `ModelBehaviorError`-style validation issues. Bad
handoff *arguments* from the model degrade gracefully instead of raising (see `input_type`
above), matching how a regular tool's bad arguments are already handled.

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
| `output_guardrails` | **Implemented** — see [Guardrails](#guardrails) |
| `tool_input_guardrails` / `tool_output_guardrails` | **Implemented** — see [Tool guardrails](#tool-guardrails) |
| `handoffs=` / `handoff()` | **Implemented** — see [Handoffs](#handoffs) |
| `agent.run(interactive=True)` | **Implemented** — see [Public methods](#public-methods) above |
| `validate_model`, `include_experimental` | Accepted, currently no-ops |
| Native provider function-calling | Not used — tool calls are parsed from prompt text via JSON convention |
