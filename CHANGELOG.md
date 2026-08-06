# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
