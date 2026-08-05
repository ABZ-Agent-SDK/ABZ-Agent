# Tools

Tools let an agent take actions or fetch information beyond what the language model knows
on its own — arithmetic, the current time, a web search, a database lookup, etc.

## How tool-calling works in this SDK

ABZ Agent SDK uses each provider's **native function/tool-calling API** when the provider
supports it (`ModelProvider.supports_native_tools` — `True` for both Gemini and Groq
today). Every registered tool's `name`, `description`, and JSON Schema (from the tool's
Pydantic `schema`, via `Tool.schema.model_json_schema()` — see `tool_to_schema()` in
`core/tools.py`) are translated into the provider's own tool-definition format and sent
structurally alongside the prompt, not as text. When the model calls a tool, it comes back
as a real field on the response (`GenerationResult.tool_calls`), extracted by
`Agent._generate_and_dispatch()` — no text parsing involved.

The SDK then, regardless of which mode produced the call:

1. Looks the tool up by name in `self.tools` (a `dict[str, Tool]`).
2. Validates/coerces the `args` dict into keyword arguments using the tool's Pydantic `schema` (`tool.parse_args`).
3. Calls `tool.run(**kwargs)`.
4. Records the result into `Memory` with role `"tool"`.
5. In single-turn mode, returns the tool's result directly as `AgentResult.content`. In multi-step mode, feeds the result back in as the next iteration's prompt context.

**Fallback (only for a provider with `supports_native_tools = False`)**: every registered
tool's `name`/`description` are listed in a `"Available TOOLS:"` manifest injected into the
system prompt, the model is instructed (via `BASE_SYSTEM_PROMPT`) to output a single-line
JSON object `{"tool": "tool_name", "args": {...}}`, and `Agent._maybe_parse_toolcall()`
regex-extracts and parses it from the raw text. This path is fully functional but not what
you'll hit with Gemini or Groq — it exists for a hypothetical future provider without
native tool-calling support.

## Import

```python
from abzagent import function_tool, Tool, ToolCall
from abzagent.core.tools import FunctionTool  # manual/OpenAI-style construction
```

## `ToolCall`

```python
@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
```

Produced internally — either directly from a provider's native tool-calling response
(`GenerationResult.tool_calls`), or, in fallback mode, by `Agent._maybe_parse_toolcall()`
parsing a tool-call JSON blob out of the model's text output. You generally don't construct
this yourself.

## `Tool` — the base interface

Every tool, however it's created, ultimately satisfies this contract:

```python
class Tool:
    name: str = "tool"
    description: str = "A tool."
    schema: Optional[Type[pydantic.BaseModel]] = None

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce a raw JSON args dict into kwargs via self.schema (pydantic v2 model_dump())."""

    def run(self, **kwargs) -> str:
        raise NotImplementedError
```

- `name` — how the model refers to the tool in its `{"tool": "..."}` calls, and the key
  used in `Agent.tools`.
- `description` — shown to the model in the tools manifest; write this so the model can
  decide *when* to call the tool.
- `schema` — an optional Pydantic model used to validate/coerce the `args` dict. If `None`,
  args pass through unvalidated.
- `run(**kwargs) -> str` — the actual implementation. Should return a string (non-string
  returns from decorator-based tools are stringified automatically, but a manual `Tool`
  subclass should return `str` itself).

### Subclassing `Tool` directly

Use this when you want full control over validation and execution:

```python
from pydantic import BaseModel
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

agent = Agent(name="Researcher", instructions="...", tools=[SearchTool()])
```

## `function_tool` — the fast path

The primary, recommended way to define a tool: decorate a plain Python function and the SDK
builds the `Tool` (name, description, Pydantic argument schema) from its signature,
type hints, and docstring.

```python
def function_tool(
    _fn: Optional[Callable] = None,
    *,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    use_docstring_info: bool = True,
) -> Tool
```

Both decorator forms are supported:

```python
@function_tool          # no parentheses
@function_tool()        # with parentheses — required when passing options

@function_tool(
    name_override="weather",
    description_override="Get the weather for a city.",
)
def get_weather(city: str) -> str: ...
```

**Important:** `function_tool` returns a `Tool` **instance** immediately (a dynamically
built `_AutoFunctionTool` subclass, already instantiated) — not the original function. If
you decorate `get_weather` and later try to call `get_weather("Paris")` directly, you're
calling the `Tool` object, not the plain function; call `get_weather.run(city="Paris")` (or
just pass the tool into `Agent(tools=[get_weather])` and let the agent invoke it).

```python
from abzagent import function_tool, Agent

@function_tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny, 24°C in {city}."

agent = Agent(
    name="WeatherBot",
    instructions="You help with weather questions.",
    tools=[get_weather],
)

result = agent.run("What's the weather in Istanbul?")
print(result.content)
```

- `name` defaults to `fn.__name__`; override with `name_override`.
- `description` defaults to `inspect.getdoc(fn)` (the function's docstring); override with `description_override`.
- Parameters named `self`, `cls`, `ctx`, or `context` are automatically excluded from the generated schema.

### Supported argument types

The signature-to-schema builder (`_model_from_signature` / `_normalize_annotation`) handles:

- Primitives: `str`, `int`, `float`, `bool`, `list`, `dict`
- `Optional[T]` / `Union[T, None]`
- `List[T]`, `Dict[K, V]`
- `Annotated[T, Field(description=...)]` — description is captured into the generated schema
- Pydantic `BaseModel` subclasses — used as-is as the field's type
- `dataclasses` — converted to a Pydantic model at runtime
- `TypedDict` — collapsed to `Dict[str, Any]` (values are **not** individually validated per key, despite the `TypedDict` shape)

Parameters without a type hint fall back to `Any` (unvalidated).

### Async tools

Fully supported — both `function_tool`-decorated async functions and `python_fn=`/`on_invoke_tool=` passed to `FunctionTool` may be coroutines:

```python
import asyncio

@function_tool
async def fetch_data(url: str) -> str:
    """Fetch content from a URL."""
    await asyncio.sleep(0)
    return f"Data from {url}"
```

Execution uses `asyncio.run()`; if already inside a running event loop, the SDK falls back
to `nest_asyncio.apply()` and `loop.run_until_complete(...)`.

### Rich argument descriptions

Two ways to give the model better per-argument guidance, in priority order (`Annotated` wins if both are present):

**1. `Annotated` + Pydantic `Field`:**

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

**2. Google-style `Args:` docstring block** (used when `use_docstring_info=True`, the default, and no `Annotated` description is present):

```python
@function_tool
def current_time(timezone: str = "UTC") -> str:
    """Get the current time in a given IANA timezone.
    Args:
        timezone: e.g. 'America/Los_Angeles', 'Asia/Karachi'
    """
    ...
```

The docstring parser (`_extract_arg_descriptions_best_effort`) is a best-effort, single-pass
line scanner — it looks for a literal `Args:` line, then reads subsequent
`name: description` or `- name: description` lines until it hits a blank line. It does not
handle multi-line descriptions or other docstring styles (NumPy, reST, etc.).

### Error handling in decorator-built tools

If the wrapped function raises, `_AutoFunctionTool.run()` catches it and re-raises as
`RuntimeError(f"Tool '{self.name}' raised error: {e}")`. `Agent._execute_tool()` in turn
catches that (and any other exception) and returns the string `"[Tool Error] {e}"` as the
tool's observation, rather than propagating the exception up through `agent.run()`. If
argument validation fails (missing/mistyped fields), the agent instead returns a
human-readable prompt like `"I need the following information to proceed: <field names>"`.

## `FunctionTool` — manual / OpenAI-schema-style construction

For cases where you want to hand-write a JSON-Schema-style parameter spec (e.g. porting
tool definitions from an OpenAI-style integration) instead of relying on signature
inspection:

```python
FunctionTool(
    *,
    name: str,
    description: str,
    params_json_schema: Optional[Dict[str, Any]] = None,
    on_invoke_tool: Optional[Callable[[Any, str], str]] = None,
    python_fn: Optional[Callable[..., Any]] = None,
)
```

Provide **either** `on_invoke_tool` (receives `(ctx, args_json: str)`, i.e. a JSON-string
handler) **or** `python_fn` (receives `**kwargs` directly) — not both. If neither is
provided, calling `run()` raises `RuntimeError(f"FunctionTool '{name}' has no implementation.")`.

`params_json_schema` supports the minimal shape `{type, properties, required, default,
description}`; only `string/integer/number/boolean/array/object` JSON types are mapped
(to `str/int/float/bool/list/dict` respectively — anything else becomes `Any`).
`additionalProperties` is set to `False` automatically unless you set it explicitly.

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
    python_fn=lambda expression: str(eval(expression)),  # use a safe evaluator in real code — see MathTool below
)
```

Or with a JSON-string handler:

```python
import json

tool = FunctionTool(
    name="echo",
    description="Echo back the input.",
    on_invoke_tool=lambda ctx, args_json: json.loads(args_json)["text"],
)
```

## Registering tools with an Agent

Three ways:

**1. At construction**, mixing `Tool` instances and plain callables:

```python
agent = Agent(name="Bot", instructions="...", tools=[MathTool(), get_weather])
```

Plain callables passed this way are auto-wrapped with `function_tool()`. If that wrapping
itself fails for some reason, the SDK falls back to a permissive `_FallbackTool` that calls
the raw function with whatever kwargs it's given (no schema validation) and stringifies the
result or catches exceptions into `"[Tool Error] {e}"`.

**2. After construction**, via `agent.register_tool(tool)`:

```python
agent.register_tool(my_tool)
```

Same coercion rules as above. Registering a tool whose `name` matches an existing one
overwrites it.

**3. Automatically**, via `agent.as_tool(...)` (wrapping a whole agent as a tool for another
agent — see [Agents docs](agents.md#agentas_tool)) — not something you call on a `Tool`
directly, but worth knowing it produces a regular `Tool` you register the same way as any other.

## Built-in tools

The SDK ships two ready-to-use tools under `abzagent.Tools` (note: capital `T`, and this
is a namespace package with no `__init__.py` — you must import the specific submodule, not
`abzagent.Tools` itself).

### `MathTool` — safe arithmetic calculator

```python
from abzagent.Tools.tools_math import MathTool, safe_eval
```

- Registered tool `name`: `"calculator"`.
- `schema`: `EvalSchema(BaseModel)` with a single required field `expression: str`.
- Implementation: `safe_eval(expr: str) -> float`, an `ast`-based whitelist evaluator — it
  parses the expression with Python's `ast` module and only permits numeric constants,
  `BinOp` (`+ - * / ** %`), and unary `+`/`-`. Anything else (names, function/attribute
  calls, imports, comparisons, lambdas, comprehensions, strings, subscripting, etc.) raises
  `ValueError`/`KeyError` rather than being evaluated.
- **This exists specifically to avoid a real CVE**: an earlier version of this tool used
  raw `eval()` and was vulnerable to arbitrary code execution (fixed in v0.3.1 — see
  `CHANGELOG.md`). Prefer `safe_eval` (or `MathTool`) over `eval()` in any tool you write
  that evaluates user-provided arithmetic.
- Note: floor division (`//`) is **not** supported — there is no `ast.FloorDiv` entry in
  `_ALLOWED_OPS`, and `tests/test_tools_math.py::test_integer_division` confirms `10 // 3`
  raises. Modulo (`%`) **is** supported — `_ALLOWED_OPS` maps `ast.Mod: operator.mod`, so
  `safe_eval("10 % 3")` returns `1`. This contradicts the repo's own
  `test_modulo` test case, which asserts `10 % 3` raises `ValueError`/`KeyError` — that test
  is stale relative to the current implementation and should be fixed or removed rather than
  taken as documentation of behavior.

```python
agent = Agent(
    name="MathBot",
    instructions="You solve arithmetic problems.",
    tools=[MathTool()],
)

result = agent.run("What is (1.5 + 2.5) * 2?")
print(result.content)   # 8.0
```

Direct use (bypassing the agent entirely):

```python
from abzagent.Tools.tools_math import safe_eval

print(safe_eval("2 + 3 * 4"))          # 14
print(safe_eval("(1.5 + 2.5) * 2"))    # 8.0
```

### `TimeTool` — current time with timezone

```python
from abzagent.Tools.tools_time import TimeTool
```

- Registered tool `name`: `"clock"`.
- `schema`: `ClockSchema(BaseModel)` with one optional field `timezone: str | None = None`.
- Implementation: returns `datetime.now(ZoneInfo(tz)).isoformat()` for the requested IANA
  timezone; if the timezone string is missing or invalid, silently falls back to UTC
  (`datetime.now(timezone.utc)`) rather than raising.

```python
agent = Agent(
    name="ClockBot",
    instructions="You tell the time.",
    tools=[TimeTool()],
)

result = agent.run("What time is it in Tokyo?")
print(result.content)
```

Direct use:

```python
tool = TimeTool()
print(tool.run(timezone="America/New_York"))   # e.g. 2026-08-01T04:23:45-04:00
print(tool.run())                              # UTC, since no timezone given
```

Accepts any [IANA timezone](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
string (`"Europe/London"`, `"Asia/Tokyo"`, `"US/Pacific"`, etc.).

### Not a reusable built-in: `Reasercher.py`

`abzagent/Tools/Reasercher.py` is a standalone demo script (a Tavily-search-backed CLI
loop), not a packaged, reusable tool module — it has no clean importable `Tool` you're meant
to drop into `tools=[...]`. Its `web_search` function also has a stray, unannotated leading
parameter (`def web_search(se, query: str, ...)`), which would produce an unintended extra
`se` field if run through `function_tool()`. Don't present this as a built-in tool in
documentation; treat it as example/reference code at most.

## Known limitations summary

| Behavior | Status |
|---|---|
| Tool discovery/invocation mechanism | Prompt-text JSON convention, not native provider function-calling |
| `TypedDict` tool arguments | Collapsed to `Dict[str, Any]` — not validated per key |
| Docstring `Args:` parsing | Best-effort, single format only (`Args:` block, one line per param) |
| `MathTool` floor division (`//`) | Not supported (raises) |
| `MathTool` modulo (`%`) | Supported and works correctly — a stale repo test (`test_modulo`) incorrectly asserts it raises |
| `Tools/Reasercher.py` | Demo script, not a reusable built-in tool |
