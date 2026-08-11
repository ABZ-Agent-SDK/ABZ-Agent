from __future__ import annotations
import os
import asyncio
import inspect
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Type, Union
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()

# silence noisy libs
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_SUPPRESS_LOGS"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .memory import Memory
from .tools import Tool, ToolCall, ToolSchema, tool_to_schema, function_tool
from .output import AgentOutputSchema, ModelBehaviorError
from .context import RunContextWrapper

# optional handoffs import
try:
    from . import handoffs as _handoffs_module
    from .handoffs import (
        Handoff,
        handoff as handoff_factory,
        HandoffError,
        CircularHandoffError,
        MaxHandoffDepthExceededError,
        HandoffInputData,
        RECOMMENDED_PROMPT_PREFIX,
        RECOMMENDED_PROMPT_PREFIX_NATIVE,
    )
    handoff = handoff_factory  # ✅ public alias
except Exception as _handoffs_import_error:
    _handoffs_module = None
    Handoff = None
    HandoffError = RuntimeError
    CircularHandoffError = RuntimeError
    MaxHandoffDepthExceededError = RuntimeError
    HandoffInputData = None
    RECOMMENDED_PROMPT_PREFIX = ""
    RECOMMENDED_PROMPT_PREFIX_NATIVE = ""
    _handoffs_err = _handoffs_import_error

    def handoff(agent):
        raise HandoffError(
            f"Handoffs module failed to import (this indicates an SDK bug): {_handoffs_err}"
        )

# optional guardrails
try:
    from .guardrails import (
        run_input_guardrails,
        run_output_guardrails,
        run_tool_input_guardrails,
        run_tool_output_guardrails,
        ToolGuardrailTripwireTriggered,
        InputGuardrail,
        OutputGuardrail,
    )
except Exception:
    def run_input_guardrails(*args, **kwargs): return None
    def run_output_guardrails(*args, **kwargs): return None
    def run_tool_input_guardrails(*args, **kwargs): return None
    def run_tool_output_guardrails(*args, **kwargs): return None
    ToolGuardrailTripwireTriggered = RuntimeError
    InputGuardrail = OutputGuardrail = None

from ..config import SDKConfig
from ..providers.base import GenerationResult
from ..providers.gemini import GeminiProvider
from ..providers.groq import GroqProvider

# Fallback-mode only (provider.supports_native_tools is False): teaches the
# JSON-blob tool-calling convention that _maybe_parse_toolcall() looks for.
BASE_SYSTEM_PROMPT = (
    "You are the ABZ Agent SDK runtime.\n"
    "When you need a TOOL, output ONLY a single JSON object on ONE line, with NO markdown/backticks/extra text:\n"
    '{"tool":"<name>","args":{...}}\n'
    "When replying to the user, DO NOT include any tool JSON—reply only with the final answer text.\n"
    "Use ONLY tool names exactly as given in the tools manifest."
)

# Native-tool-calling mode: no JSON-blob convention to teach — the provider's
# own tool-calling mechanism handles that structurally.
BASE_SYSTEM_PROMPT_NATIVE = "You are the ABZ Agent SDK runtime."

InstructionsFn = Callable[[RunContextWrapper, "Agent"], Union[str, Awaitable[str]]]


class AgentResult:
    def __init__(
        self,
        content: str,
        steps: List[str],
        parsed: Any = None,
        last_agent: Optional["Agent"] = None,
    ):
        self.content = content
        self.steps = steps
        self.parsed = parsed
        self.last_agent = last_agent


class Agent:
    def __init__(
        self,
        *,
        name: str,
        instructions: Union[str, InstructionsFn],
        model: Optional[str] = "gemini-2.0-flash",
        tools: Optional[List[Union[Tool, Callable[..., Any]]]] = None,
        handoffs: Optional[List[Union["Agent", Handoff]]] = None,
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
    ) -> None:
        if not name:
            raise ValueError("Agent 'name' is required.")
        if not instructions or (isinstance(instructions, str) and not instructions.strip()):
            raise ValueError("Agent 'instructions' is required (string or function).")

        self.name = name
        self._instructions_src = instructions
        self.instructions: str = ""
        self.verbose = verbose
        self.max_iterations = max_iterations if max_iterations is not None else 1
        self.memory = memory or Memory()
        self.output_type = output_type
        self._output_schema: Optional[AgentOutputSchema] = (
            AgentOutputSchema(output_type) if output_type is not None else None
        )
        self.input_guardrails = [
            InputGuardrail(g) if isinstance(g, str) else g for g in (input_guardrails or [])
        ]
        self.output_guardrails = [
            OutputGuardrail(g) if isinstance(g, str) else g for g in (output_guardrails or [])
        ]
        self.tool_input_guardrails = list(tool_input_guardrails or [])
        self.tool_output_guardrails = list(tool_output_guardrails or [])

        self.model = self._resolve_model_param(
            model=model,
            include_experimental=include_experimental,
            validate_model=validate_model,
        )

        # tools
        self.tools = self._normalize_and_index_tools(tools or [])

        # handoffs
        self._handoffs: List[Handoff] = []
        for item in (handoffs or []):
            if Handoff is not None and isinstance(item, Handoff):
                self._handoffs.append(item)
            else:
                self._handoffs.append(handoff_factory(agent=item))
        for h in self._handoffs:
            ht = h.to_tool(self)
            self.tools[ht.name] = ht

        # Detect provider based on model name
        provider_type = SDKConfig.detect_provider(self.model)
        
        cfg_env = SDKConfig()
        resolved_key = api_key
        
        # If no key provided, try to get from environment based on provider
        if not resolved_key:
            if provider_type == "groq":
                resolved_key = os.getenv("GROQ_API_KEY", "")
            else:
                resolved_key = os.getenv("GEMINI_API_KEY", "")
        
        if not resolved_key:
            key_name = "GROQ_API_KEY" if provider_type == "groq" else "GEMINI_API_KEY"
            raise RuntimeError(f"{key_name} missing — set in env or .env")

        cfg = SDKConfig(
            model=self.model,
            api_key=resolved_key,
            provider=provider_type,
            temperature=cfg_env.temperature,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )
        
        # Dynamically select provider
        if provider_type == "groq":
            self.provider = GroqProvider(cfg)
        else:
            self.provider = GeminiProvider(cfg)

    # ---------------- PUBLIC ----------------

    def register_tool(self, tool: Union[Tool, Callable[..., Any]]) -> None:
        t = self._coerce_tool(tool, idx=-1)
        self.tools[t.name] = t

    class _AgentInvokeSchema(BaseModel):
        message: str

    def as_tool(self, *, tool_name: str, tool_description: str) -> Tool:
        outer = self

        class _AgentTool(Tool):
            name = tool_name
            description = tool_description
            schema = Agent._AgentInvokeSchema

            def run(self, **kwargs) -> str:
                msg = kwargs.get("message", "Please take over from here.")
                return outer.run(msg).content

        return _AgentTool()

    # ---------------- run() ----------------
    def run(
        self,
        user_message: Optional[str] = None,
        *,
        context: Any = None,
        interactive: bool = False,
        _interactive: bool = False,
    ) -> Optional[AgentResult]:
        """
        The SDK's single execution entrypoint.

        - `agent.run("Hello")` — single request, returns an AgentResult (unchanged).
        - `agent.run(interactive=True)` — starts an interactive terminal REPL that
          calls this same method for every message; returns None when the session ends.
        """
        if interactive:
            if user_message is not None:
                raise ValueError(
                    "Agent.run() cannot take both a user_message and interactive=True."
                )
            self._run_interactive(context=context)
            return None

        if user_message is None:
            raise ValueError("Agent.run() requires a user_message when interactive=False.")

        return self._run(user_message, context=context, _handoff_path=[], _interactive=_interactive)

    def _run_interactive(self, *, context: Any = None) -> None:
        """
        Interactive terminal loop. A thin wrapper around run() — no inference
        logic lives here, so every feature (memory, tools, structured output,
        handoffs, future guardrails) works automatically since each turn is
        just a normal self.run(user_input) call.
        """
        print(f"🤖 {self.name} started. Type 'exit' to quit.")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                return

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                return

            try:
                result = self.run(user_input, context=context, _interactive=True)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
            except Exception as e:
                print(f"[Error] {e}")
                continue

            responder = result.last_agent.name if result.last_agent is not None else self.name
            print(f"{responder}: {result.content}")

    def _run(
        self,
        user_message: str,
        *,
        context: Any = None,
        _handoff_path: Optional[List["Agent"]] = None,
        _interactive: bool = False,
    ) -> AgentResult:
        steps: List[str] = []
        ctx_for_run = RunContextWrapper(
            current_agent=self,
            target_agent=self,
            memory=self.memory,
            steps=steps,
            context=context,
        )

        if self.input_guardrails:
            run_input_guardrails(
                guards=self.input_guardrails,
                ctx=ctx_for_run,
                agent=self,
                user_input=user_message,
            )

        self.memory.remember("user", user_message)

        # Single-turn mode
        if self.max_iterations <= 1:
            effective_instructions = self._resolve_instructions(ctx_for_run)
            prompt = self._build_prompt(user_message, effective_instructions=effective_instructions)
            result = self._generate_and_dispatch(prompt)

            tool_call = result.tool_calls[0] if result.tool_calls else None
            if tool_call:
                call_repr = json.dumps({"tool": tool_call.tool, "args": tool_call.args})
                steps.append(call_repr)
                tool = self.tools.get(tool_call.tool)
                if tool is not None and getattr(tool, "is_handoff", False):
                    return self._perform_handoff(
                        tool, tool_call, steps=steps, context=context,
                        _handoff_path=_handoff_path, _interactive=_interactive,
                    )
                obs = self._execute_tool(tool_call, ctx=ctx_for_run)
                self.memory.remember("tool", obs)
                return AgentResult(content=obs, steps=[call_repr, obs], last_agent=self)

            model_out = self._normalize_output(result.text)
            steps.append(model_out)
            final_text, parsed = self._resolve_output(model_out, base_prompt=prompt, steps=steps)
            self._check_output_guardrails(parsed if parsed is not None else final_text, ctx=ctx_for_run)
            return AgentResult(content=final_text, steps=steps, parsed=parsed, last_agent=self)

        # Iterative mode
        for _i in range(self.max_iterations):
            effective_instructions = self._resolve_instructions(ctx_for_run)
            # On the last available iteration, don't offer tools — force a
            # plain-text answer instead of letting the model spend its final
            # turn on yet another tool call (which would exhaust the budget
            # with nothing to show for it; see "Reached iteration limit"
            # below). This guarantees any max_iterations >= 2 setup ends in a
            # real answer for a normal one-or-more-tool-call workflow, even
            # if an earlier turn called the wrong tool or re-requested one.
            is_last_iteration = _i == self.max_iterations - 1
            prompt = self._build_prompt(
                user_message if _i == 0 else "Continue.",
                effective_instructions=effective_instructions,
                allow_tools=not is_last_iteration,
            )

            result = self._generate_and_dispatch(prompt, allow_tools=not is_last_iteration)

            tool_call = result.tool_calls[0] if result.tool_calls else None
            if tool_call:
                call_repr = json.dumps({"tool": tool_call.tool, "args": tool_call.args})
                steps.append(call_repr)
                tool = self.tools.get(tool_call.tool)
                if tool is not None and getattr(tool, "is_handoff", False):
                    return self._perform_handoff(
                        tool, tool_call, steps=steps, context=context,
                        _handoff_path=_handoff_path, _interactive=_interactive,
                    )
                obs = self._execute_tool(tool_call, ctx=ctx_for_run)
                self.memory.remember("tool", obs)
                user_message = f"TOOL RESULT ({tool_call.tool}): {obs}"
                continue

            model_out = self._normalize_output(result.text)
            steps.append(model_out)
            final_text, parsed = self._resolve_output(model_out, base_prompt=prompt, steps=steps)
            self._check_output_guardrails(parsed if parsed is not None else final_text, ctx=ctx_for_run)
            return AgentResult(content=final_text, steps=steps, parsed=parsed, last_agent=self)

        fallback = "Reached iteration limit without final answer.\n\n" + (steps[-1] if steps else "")
        self._check_output_guardrails(fallback, ctx=ctx_for_run)
        return AgentResult(content=fallback, steps=steps, last_agent=self)

    def _perform_handoff(
        self,
        tool: Tool,
        call: ToolCall,
        *,
        steps: List[str],
        context: Any,
        _handoff_path: Optional[List["Agent"]],
        _interactive: bool = False,
    ) -> AgentResult:
        handoff_obj: Handoff = tool.handoff
        target: "Agent" = handoff_obj.target_agent
        path = [*(_handoff_path or []), self]

        max_depth = _handoffs_module.MAX_HANDOFF_DEPTH if _handoffs_module is not None else 5

        if target in path:
            chain = " -> ".join(a.name for a in path) + f" -> {target.name}"
            raise CircularHandoffError(f"Circular handoff detected: {chain}")
        if len(path) > max_depth:
            chain = " -> ".join(a.name for a in path)
            raise MaxHandoffDepthExceededError(
                f"Handoff chain exceeded MAX_HANDOFF_DEPTH={max_depth}: {chain}"
            )

        try:
            kwargs = tool.parse_args(call.args)
        except ValueError as ve:
            obs = self._format_validation_error(ve)
            self.memory.remember("tool", obs)
            return AgentResult(content=obs, steps=[*steps, obs], last_agent=self)

        # Snapshot conversation history *before* recording the transfer breadcrumb,
        # so that breadcrumb doesn't leak into the target's replayed copy.
        input_data = HandoffInputData(messages=list(self.memory.load()))
        if handoff_obj.input_filter is not None:
            input_data = handoff_obj.input_filter(input_data)

        if target.memory is not self.memory:
            for m in input_data.messages:
                target.memory.remember(m.role, m.content)

        self.memory.remember("tool", f"Transferred conversation to '{target.name}'.")

        input_instance = handoff_obj.input_type(**kwargs) if handoff_obj.input_type is not None else None

        if handoff_obj.on_handoff is not None:
            ctx_for_handoff = RunContextWrapper(
                current_agent=self,
                target_agent=target,
                memory=self.memory,
                steps=steps,
                context=context,
            )
            try:
                if input_instance is not None:
                    handoff_obj.on_handoff(ctx_for_handoff, input_instance)
                else:
                    handoff_obj.on_handoff(ctx_for_handoff)
            except Exception as e:
                raise HandoffError(f"on_handoff callback for '{target.name}' raised: {e}") from e

        continuation = kwargs.get("message") or (
            f"You are now handling this conversation, transferred from '{self.name}'. "
            "Review the conversation above and continue helping the user."
        )

        if _interactive:
            print(f"\n🔄 Handoff\n\n{self.name}\n   │\n   ▼\n{target.name}\n")

        nested = target._run(
            continuation, context=context, _handoff_path=path, _interactive=_interactive
        )
        nested.steps = [*steps, f"[HANDOFF] {self.name} -> {target.name}", *nested.steps]
        return nested

    # ---------------- INTERNAL HELPERS ----------------
    def _resolve_output(self, model_out: str, *, base_prompt: str, steps: List[str]):
        """
        If `output_type` is set, validate the model's final text against it,
        retrying with the validation error fed back to the model a bounded
        number of times before raising ModelBehaviorError. No-op (parsed=None)
        when no output_type was declared.
        """
        if self._output_schema is None:
            return model_out, None

        current_text = model_out
        last_error: Optional[ModelBehaviorError] = None

        for attempt in range(self._output_schema.max_retries + 1):
            try:
                parsed = self._output_schema.validate_json(current_text)
                return current_text, parsed
            except ModelBehaviorError as e:
                last_error = e
                if attempt >= self._output_schema.max_retries:
                    break
                repair_prompt = (
                    f"{base_prompt}\n\n"
                    f"[YOUR PREVIOUS RESPONSE]: {current_text}\n"
                    f"[VALIDATION ERROR]: {e}\n"
                    "Respond again with ONLY corrected JSON matching the required "
                    "schema. No markdown, no explanation — JSON only."
                )
                raw = self.provider.generate(
                    repair_prompt, output_schema=self._output_schema, strict=not bool(self.tools)
                )
                current_text = self._normalize_output(raw.text)
                steps.append(current_text)

        raise last_error

    def _normalize_output(self, raw_out):
        if isinstance(raw_out, dict):
            return raw_out.get("content") or raw_out.get("text") or raw_out.get("message") or json.dumps(raw_out)
        elif hasattr(raw_out, "content"):
            return getattr(raw_out, "content")
        else:
            return str(raw_out)

    def _normalize_and_index_tools(self, tools_in: List[Union[Tool, Callable[..., Any]]]) -> Dict[str, Tool]:
        tools_dict: Dict[str, Tool] = {}
        for idx, raw in enumerate(tools_in):
            if isinstance(raw, Tool):
                name = getattr(raw, "name", None) or f"tool_{idx}"
                tools_dict[name] = raw
                continue
            if callable(raw):
                try:
                    wrapped = function_tool()(raw)
                    name = getattr(wrapped, "name", None) or getattr(raw, "__name__", f"tool_{idx}")
                    tools_dict[name] = wrapped
                    continue
                except Exception:
                    fn_name = getattr(raw, "__name__", f"tool_{idx}")

                    class _FallbackTool(Tool):
                        name = fn_name
                        description = f"Auto-wrapped tool for {fn_name}"
                        schema = None
                        def run(self, **kwargs) -> str:
                            try:
                                out = raw(**kwargs)
                                return out if isinstance(out, str) else str(out)
                            except Exception as e:
                                return f"[Tool Error] {e}"

                    tools_dict[_FallbackTool.name] = _FallbackTool()
        return tools_dict

    def _resolve_instructions(self, ctx: RunContextWrapper) -> str:
        src = self._instructions_src
        if isinstance(src, str):
            self.instructions = src
            return src
        fn = src
        if inspect.iscoroutinefunction(fn):
            try:
                return asyncio.run(fn(ctx, self))
            except RuntimeError:
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(fn(ctx, self))
        else:
            text = fn(ctx, self)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Dynamic instructions function must return non-empty string.")
            self.instructions = text
            return text

    def _execute_tool(self, call: ToolCall, *, ctx: RunContextWrapper) -> str:
        tool = self.tools.get(call.tool)
        if not tool:
            return f"[Tool Error] Unknown tool: {call.tool}"

        try:
            kwargs = tool.parse_args(call.args)
        except ValueError as ve:
            return self._format_validation_error(ve)

        if self.tool_input_guardrails:
            blocked = run_tool_input_guardrails(
                guards=self.tool_input_guardrails, ctx=ctx, agent=self, call=call, kwargs=kwargs,
            )
            if blocked is not None:
                return blocked

        try:
            if self.verbose:
                print(f"Executing tool {call.tool} with kwargs={kwargs}")
            output = tool.run(**kwargs)
        except Exception as e:
            return f"[Tool Error] {e}"

        if self.tool_output_guardrails:
            blocked = run_tool_output_guardrails(
                guards=self.tool_output_guardrails, ctx=ctx, agent=self, call=call, kwargs=kwargs, output=output,
            )
            if blocked is not None:
                return blocked

        return output

    def _check_output_guardrails(self, final_output: Any, *, ctx: RunContextWrapper) -> None:
        if self.output_guardrails:
            run_output_guardrails(
                guards=self.output_guardrails, ctx=ctx, agent=self, final_output=final_output,
            )

    def _format_validation_error(self, ve: ValueError) -> str:
        # Tool.parse_args() wraps pydantic's ValidationError in a plain ValueError
        # (`raise ValueError(...) from e`); unwrap it if present to get the nice
        # per-field message, otherwise fall back to the raw error text.
        source = ve if hasattr(ve, "errors") else getattr(ve, "__cause__", None)
        if source is not None and hasattr(source, "errors"):
            missing_fields = []
            for err in source.errors():
                loc = err.get("loc", ["unknown"])
                typ = err.get("type")
                if typ == "missing":
                    missing_fields.append(loc[0])
                elif typ.endswith("_parsing"):
                    missing_fields.append(f"{loc[0]} (invalid type)")
            if missing_fields:
                return f"I need the following information to proceed: {', '.join(missing_fields)}"
        return f"Invalid arguments: {ve}"

    def _build_prompt(
        self, user_message: str, *, effective_instructions: str, allow_tools: bool = True
    ) -> str:
        native = self.provider.supports_native_tools
        # When this turn isn't allowed to offer tools (see allow_tools below),
        # there's nothing to teach the model about calling one — use the
        # plain base prompt (same one native-mode already uses) regardless
        # of provider, so a fallback-mode model isn't still told "output a
        # tool-call JSON blob" on a turn where no tool schemas are provided.
        base = BASE_SYSTEM_PROMPT_NATIVE if (native or not allow_tools) else BASE_SYSTEM_PROMPT
        system = (
            f"{base}\n\n"
            f"[AGENT NAME]: {self.name}\n"
            f"[INSTRUCTIONS]: {effective_instructions}\n"
            f"[MODEL]: {self.model}\n"
        )
        # In native mode, tools are advertised to the provider structurally
        # (see _generate_and_dispatch) — a text manifest would be redundant.
        if self.tools and not native and allow_tools:
            manifest = ["Available TOOLS:"]
            for n, t in self.tools.items():
                desc = (t.description or "").strip().replace("\n", " ")
                manifest.append(f"- {n}: {desc}")
            system += "\n" + "\n".join(manifest)

        if self._output_schema is not None and not self._output_schema.is_plain_text:
            system += "\n\n" + self._output_schema.prompt_instructions()

        history = self.memory.to_prompt()

        # The handoff hint is deliberately the LAST thing before the user's
        # message, not folded into `system` above — models weight the end of
        # the prompt more heavily, and this is the single highest-leverage
        # reminder for getting weaker models to actually transfer instead of
        # answering directly.
        tail = ""
        if self._handoffs and allow_tools:
            tail = "\n\n" + (RECOMMENDED_PROMPT_PREFIX_NATIVE if native else RECOMMENDED_PROMPT_PREFIX)

        return f"[SYSTEM]: {system}\n\n{history}{tail}\n\n[USER]: {user_message}"

    def _generate_and_dispatch(self, prompt: str, *, allow_tools: bool = True) -> GenerationResult:
        """
        Calls the provider once and returns a GenerationResult with either
        `.text` or `.tool_calls` populated — regardless of whether the
        provider used native tool calling or the prompt+regex fallback.
        Callers (both _run() branches) never need to know which mode
        produced the result.

        `allow_tools=False` (used for the forced-final iteration of the
        iterative loop, see _run()) omits tool schemas from the request
        entirely — for native providers this makes it structurally
        impossible for the model to request a tool on this call, guaranteeing
        a plain-text response instead of another tool-call attempt.
        """
        native = self.provider.supports_native_tools
        offer_tools = allow_tools and bool(self.tools)
        tool_schemas = [tool_to_schema(t) for t in self.tools.values()] if (native and offer_tools) else None

        result = self.provider.generate(
            prompt,
            tools=tool_schemas,
            output_schema=self._output_schema,
            strict=not offer_tools,
        )

        if not native and offer_tools and not result.tool_calls and result.text:
            tool_call = self._maybe_parse_toolcall(result.text)
            if tool_call:
                return GenerationResult(tool_calls=[tool_call])

        return result

    def _maybe_parse_toolcall(self, text: str) -> Optional[ToolCall]:
        blob = _extract_json_blob(text.strip())
        if not blob:
            return None
        try:
            data = json.loads(blob)
            if isinstance(data, dict) and "tool" in data:
                args = data.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                return ToolCall(tool=str(data["tool"]), args=args)
        except Exception:
            return None
        return None

    def _resolve_model_param(self, *, model: Optional[str], include_experimental: bool, validate_model: bool) -> str:
        if not model or str(model).strip().lower() == "auto":
            return "gemini-2.0-flash"
        return str(model)


_JSON_OBJECT_OR_ARRAY = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
def _extract_json_blob(text: str) -> Optional[str]:
    m = _JSON_OBJECT_OR_ARRAY.search(text.strip())
    if not m:
        return None
    return m.group(1)