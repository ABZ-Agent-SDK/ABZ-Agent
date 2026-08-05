# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
