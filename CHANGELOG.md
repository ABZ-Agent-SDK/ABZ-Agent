# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.7] - 2026-08-13

### Fixed

- **The model catalog was untrustworthy — some of it was never verified, and one entry
  (the SDK's own default) was completely dead.** `GeminiModel`/`GroqModel`
  (`abzagent.providers.model_types`) previously included models — DeepSeek among them —
  that Groq doesn't even serve, inherited uncritically from a list that predates this
  package. Rebuilt both from direct calls to each provider's real `/models` endpoint plus
  an actual `generate`/`chat.completions.create` call per candidate, not docs scraping or
  memory. Two Gemini models that are listed but 404 with "no longer available to new
  users" for a real key (`gemini-2.5-pro`, `gemini-2.5-flash-lite`) were deliberately
  excluded — that message means genuinely blocked for new callers, not a quota/tier 429.
  Separately, and more seriously: `"gemini-2.0-flash"`, the SDK's own default `model=`
  value, was fully retired by Google — every `Agent(...)` call omitting `model=` was
  silently broken. Default is now `"gemini-2.5-flash"`, fixed everywhere it appeared
  (`Agent.__init__`, `_resolve_model_param`, `SDKConfig`, the natural-language guardrails
  classifier's fast-model default, the `abz-agents setup` CLI wizard).
- **`SDKConfig.detect_provider()` misrouted newer Groq model names to Gemini.** Its
  substring patterns (`"qwen/"`, `"llama"`, `"mixtral"`, ...) don't match newer model
  name shapes like `"openai/gpt-oss-120b"` or `"groq/compound"`. Added an exact-match
  fast path against the live-verified Groq model list, checked before the substring
  fallback, so these route correctly without weakening the fallback for genuinely
  custom/future models.
- **`groq_catalog`/`gemini_catalog`'s `best_default()` referenced models that no longer
  exist,** working only by fallback coincidence. Candidate lists now reference real,
  live-verified models for each speed/quality/balanced tier.

### Added

- **`python scripts/update_model_catalog.py`** — calls both providers' live `/models`
  endpoints and reports drift against the current `GeminiModel`/`GroqModel` Literals.
  `Literal[...]` members must be static source text for IDE autocomplete to see them, so
  this doesn't edit the catalog automatically; it's the documented way to check whether
  it needs updating.

## [0.5.6] - 2026-08-12

### Added

- **IDE autocomplete for `Agent(model=...)`.** New `abzagent.providers.model_types`
  (`GeminiModel`, `GroqModel`, `KnownModel` — re-exported from `abzagent` and
  `abzagent.core`) provides `Literal[...]` types listing each provider's documented
  model ids. `Agent.__init__`'s `model` parameter is now typed
  `Optional[Union[KnownModel, str]]`, so editors with type checking (Pylance/Pyright,
  mypy) suggest known model ids while typing `model="`, without losing the ability to
  pass any string — a new model release, a fine-tuned deployment, anything not yet
  listed all continue to work exactly as before. Pure typing aid: nothing is validated
  or enforced at runtime, and `SDKConfig.detect_provider()`'s model-string routing is
  unchanged. `abzagent.providers.groq_catalog._GROQ_MODELS` now derives from
  `GroqModel` via `typing.get_args()` instead of a separately maintained duplicate
  list, so the two can't drift apart.

## [0.5.5] - 2026-08-11

### Fixed

- **Iterative mode (`max_iterations > 1`) could exhaust its entire budget on
  tool calls and never produce a final answer.** If the model spent every
  iteration requesting a tool (e.g. repeatedly calling a misremembered tool
  name), the loop fell through to an unhelpful `"Reached iteration limit
  without final answer"` message showing the last raw tool-call JSON. The
  loop's last iteration now omits tool schemas entirely, so for native
  tool-calling providers (Gemini, Groq) it's structurally impossible for
  that turn to request a tool — guaranteeing a real natural-language answer
  for any `max_iterations >= 2` setup, regardless of how many earlier turns
  were wasted on tool calls. A well-behaved run (tool call, then answer)
  never reaches the changed code path and is completely unaffected.

## [0.5.4] - 2026-08-07

### Added

- **Natural-language guardrails**: `input_guardrails=["Block mathematical questions."]` —
  pass a plain-English policy string and the SDK runs an LLM classifier behind the
  scenes instead of you writing classification code yourself. `output_guardrails=[...]`
  works the same way, classifying the agent's final answer (or its parsed structured
  output, if `output_type` is set). For reuse across agents or to pick a specific
  classification model, use the explicit `InputGuardrail("policy", model=..., api_key=...)`
  / `OutputGuardrail(...)` factories — both now exported from `abzagent`. By default the
  classifier runs on a fast/cheap model on the *same provider* as the host agent, so no
  extra API key is required. Bare strings, `InputGuardrail(...)`/`OutputGuardrail(...)`,
  and existing `@input_guardrail`/`@output_guardrail`-decorated functions can be freely
  mixed in the same list — fully backward compatible, no constructor signature changes.
- New `abzagent/providers/factory.py` (`resolve_provider()`) — internal helper the
  classifier uses to build a standalone provider instance for a given model.

## [0.5.3] - 2026-08-06

### Fixed

- `GuardrailFunctionOutput.output_info` now defaults to `None` instead of
  being a required field — `GuardrailFunctionOutput(tripwire_triggered=False)`
  no longer raises a `pydantic.ValidationError` when there's nothing to report.
- Guardrail functions may now omit the `agent` parameter from their signature
  across all four guardrail kinds — `def my_guardrail(ctx, user_input): ...`
  works the same as the full `def my_guardrail(ctx, agent, user_input): ...`.
  Previously, a guardrail written without `agent` failed with a confusing
  "takes N positional arguments but N+1 were given" error.

## [0.5.2] - 2026-08-06

### Added

- `GuardrailResult` and `Guardrail` are now importable from `abzagent` (and
  `abzagent.core`) as aliases: `GuardrailResult` is the same class as
  `GuardrailFunctionOutput` (the canonical name used throughout the docs and
  tests), and `Guardrail` is the same class as the guardrail-wrapper type
  produced by `@input_guardrail`/`@output_guardrail`/etc. Neither name
  existed anywhere in the codebase before this release — this adds them as
  aliases rather than renaming the originals, so nothing already using
  `GuardrailFunctionOutput` needs to change.

## [0.5.1] - 2026-08-06

### Fixed

- Guardrail symbols (`input_guardrail`, `output_guardrail`, `tool_input_guardrail`,
  `tool_output_guardrail`, `GuardrailFunctionOutput`, and the three
  `*TripwireTriggered` exceptions) are now importable directly from the top-level
  `abzagent` package (`from abzagent import input_guardrail`), matching how
  Handoffs are already exported. They shipped in 0.5.0 working correctly via
  `abzagent.core.guardrails` / `abzagent.core`, but were missing from
  `abzagent/__init__.py`, so the top-level import raised `ImportError`.

## [0.5.0] - 2026-08-06

### Added

- **Output guardrails are now actually enforced.** `output_guardrails=[...]` previously had
  no effect; `Agent.run()` now runs them at every return point (single-turn, iterative, and
  the iteration-limit fallback) and raises `OutputGuardrailTripwireTriggered` on a trip.
- **Tool guardrails**: new `tool_input_guardrails=[...]` / `tool_output_guardrails=[...]`
  on `Agent(...)`, with `@tool_input_guardrail` / `@tool_output_guardrail` decorators. A
  tripped tool guardrail degrades gracefully by default (the model gets a
  `"[Tool Guardrail Blocked] ..."` substitution instead of the tool's real result) — raise
  `ToolGuardrailTripwireTriggered` yourself from inside a guardrail if you want a trip to
  hard-abort the run instead.

### Changed

- Router prompt hardened for smaller/free models: agents with `handoffs=[...]` are now
  explicitly told not to ask the user for permission before transferring, and to pick the
  single best-matching specialist instead of hedging when more than one could apply.
- Auto-derived handoff tool descriptions no longer echo a redundant "You are `<Name>`."
  persona clause from the target agent's instructions, and truncate on a word boundary
  instead of a hard character cut.
- `agent.run(interactive=True)` now labels each response with the agent that actually
  produced it (`result.last_agent.name`) instead of always the agent the loop was started
  on — a handoff's specialist is credited by name, right below its `🔄 Handoff` diagram.
- Fixed a bug where every output guardrail ran twice per check.

## [0.4.0] - 2026-08-05

### Changed

- Handoff prompting redesigned to significantly improve reliability, especially
  on smaller/free models that previously answered directly instead of
  transferring. The injected prompt now frames the agent explicitly as a
  router ("if a specialist tool clearly matches, call it — do not answer
  yourself") instead of permissive "you can delegate..." language, and is
  positioned as the last thing before the user's message rather than buried
  mid-prompt.
- Handoff tool descriptions are now generated automatically from each target
  agent's own `instructions` (no new config — `Agent(name=..., instructions=...)`
  used bare in `handoffs=[...]` now produces a real "use this agent when..."
  hint instead of a generic one-liner). `handoff(target,
  tool_description_override=...)` still overrides it when set.

### Added

- `agent.run(interactive=True)` now prints a small diagram (`🔄 Handoff`) for
  each transfer as it happens, including multi-hop chains — handled entirely
  by the SDK, nothing to print yourself. This never happens during a plain
  `agent.run("...")` call.

## [0.3.0] - 2026-08-05

### Changed

- Tool calls (including Handoffs) are now detected via each provider's native
  function/tool-calling API instead of asking the model to emit a JSON blob in
  plain text and regex-parsing it out. Both Gemini and Groq support this
  natively and use it by default — this is significantly more reliable than
  the old text convention. The old prompt+regex approach is kept as a working
  fallback for a hypothetical future provider without native tool-calling
  support, not removed; no application code changes are required to benefit
  from this.

## [0.2.0] - 2026-08-05

### Added

- **Handoffs**: pass `handoffs=[...]` (bare `Agent` instances, or `handoff()` for
  customization) to `Agent(...)` to let one agent transfer a conversation to a
  specialized agent. Routing, memory/context transfer, tool usage, and structured
  output all continue automatically in the target agent. New `AgentResult.last_agent`
  reports which agent actually produced the final answer. New exceptions
  `CircularHandoffError`, `MaxHandoffDepthExceededError`, `InvalidHandoffTargetError`.

### Changed

- `agent.chat()` has been removed. The SDK now exposes a single execution method:
  `agent.run(interactive=True)` starts the same interactive terminal loop.
  `agent.run("...")` is unchanged.

## [0.1.0] - 2026-08-04

### Changed

- Package renamed from `abagentsdk` to `abz-agents` — install with
  `pip install abz-agents`, import with `from abzagent import ...`, and use
  the CLI as `abz-agents <command>`.

### Added

- **Structured output**: pass `output_type=` (a Pydantic model, dataclass,
  `TypedDict`, or plain type) to `Agent(...)` and get a validated Python object
  back on `AgentResult.parsed`, with no manual JSON prompting or parsing.
  Uses each provider's native structured-output support where available
  (Gemini) and automatically retries once on validation failure before
  raising `ModelBehaviorError`.

## [0.3.1] - 2026-07-28

### Security

**Impact:** Conditional arbitrary code execution. This issue affects applications that explicitly register the built-in calculator tool and allow attacker-controlled input to reach `Agent.run()`. In this configuration, unrestricted use of `eval()` in the calculator tool could allow execution of arbitrary Python expressions with the privileges of the hosting application.

**Fix:** The calculator implementation has been replaced with a secure, standard-library `ast`-based arithmetic evaluator. It accepts only numeric constants, parentheses, unary `+`/`-`, and the arithmetic operators `+`, `-`, `*`, `/`, `//`, `%`, and `**`. All other AST node types—including function calls, imports, attribute access, names, lambdas, comprehensions, and comparisons—are rejected with a `ValueError`.

The calculator's public API and response format remain unchanged (`Result: ...` / `Error: ...`), so no application changes are required beyond upgrading.

**Recommendation:** Users who have registered the built-in calculator tool should upgrade to this release immediately.
