# abzagent/core/tools.py
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Type,
    Union,
    get_type_hints,
    get_origin,
    get_args,
    List,
    Tuple,
)

from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field, ValidationError, create_model


# ──────────────────────────────────────────────────────────────────────────────
# Public data structures used by Agent runtime
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]


class Tool:
    """
    Base Tool interface used by the Agent.
    Subclasses (or wrappers) must implement:
      - name: str
      - description: str
      - schema: Optional[Type[pydantic.BaseModel]] (argument model)
      - run(**kwargs) -> str
    """
    name: str = "tool"
    description: str = "A tool."
    schema: Optional[Type[BaseModel]] = None

    def parse_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce JSON args to kwargs via Pydantic if a schema is present."""
        if self.schema is None:
            return args
        try:
            model = self.schema(**args)
            return model.model_dump()  # pydantic v2
        except ValidationError as e:
            raise ValueError(f"Invalid arguments for tool '{self.name}': {e}") from e

    def run(self, **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# FunctionTool (manual construction, OpenAI-style)
# ──────────────────────────────────────────────────────────────────────────────

class FunctionTool(Tool):
    """
    Wrap a function as a Tool.

    Provide either:
      - on_invoke_tool(ctx, args_json) -> str   (JSON handler; OpenAI-style)
      - python_fn(**kwargs) -> str              (direct Python callable)

    If params_json_schema is provided, a Pydantic model is generated to validate args.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        params_json_schema: Optional[Dict[str, Any]] = None,
        on_invoke_tool: Optional[Callable[[Any, str], str]] = None,
        python_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self._on_invoke_json = on_invoke_tool
        self._python_fn = python_fn

        if params_json_schema is not None:
            # be strict by default unless user explicitly opts out
            if "additionalProperties" not in params_json_schema:
                params_json_schema["additionalProperties"] = False
            self.schema = _pydantic_model_from_json_schema(self.name, params_json_schema)
        else:
            self.schema = None

    def run(self, **kwargs) -> str:
        # 1) JSON handler
        if self._on_invoke_json is not None:
            args_json = json.dumps(kwargs, ensure_ascii=False)
            return _call_maybe_async(self._on_invoke_json, None, args_json)

        # 2) direct callable
        if self._python_fn is not None:
            return _call_maybe_async(self._python_fn, **kwargs)

        raise RuntimeError(f"FunctionTool '{self.name}' has no implementation.")


# ──────────────────────────────────────────────────────────────────────────────
# function_tool decorator — supports @function_tool and @function_tool()
# (Pylance-safe: set class attrs AFTER class body)
# ──────────────────────────────────────────────────────────────────────────────

def function_tool(
    _fn: Optional[Callable[..., Any]] = None,
    *,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    use_docstring_info: bool = True,
) -> Union[Tool, Callable[[Callable[..., Any]], Tool]]:
    """
    Turn a plain Python function into a Tool.

    Supports both:
        @function_tool
        @function_tool(...)

    Returns a Tool instance (not a callable function).
    """

    def _wrap(fn: Callable[..., Any]) -> Tool:
        tool_name = name_override or fn.__name__
        tool_desc = (description_override or (inspect.getdoc(fn) or "")).strip()

        arg_model = _model_from_signature(
            fn, use_docstring_info=use_docstring_info, tool_name=tool_name
        )

        # define class first, then set attrs (so Pylance doesn't flag undefined names)
        class _AutoFunctionTool(Tool):
            name: str = "tool"
            description: str = ""
            schema: Optional[Type[BaseModel]] = None

            def run(self, **kwargs) -> str:
                parsed = self.parse_args(kwargs)
                try:
                    return _call_maybe_async(fn, **parsed)
                except Exception as e:
                    raise RuntimeError(f"Tool '{self.name}' raised error: {e}") from e

        _AutoFunctionTool.name = tool_name
        _AutoFunctionTool.description = tool_desc
        _AutoFunctionTool.schema = arg_model

        return _AutoFunctionTool()

    # support both forms
    if _fn is None:
        return _wrap
    return _wrap(_fn)


# ──────────────────────────────────────────────────────────────────────────────
# Internals — schema building, docstring parsing, async glue
# ──────────────────────────────────────────────────────────────────────────────

def _call_maybe_async(fn: Callable[..., Any], *args, **kwargs) -> str:
    """Call sync/async fn and coerce non-str returns into str."""
    if inspect.iscoroutinefunction(fn):
        try:
            return asyncio.run(fn(*args, **kwargs))
        except RuntimeError:
            # already in an event loop
            try:
                import nest_asyncio  # type: ignore
                nest_asyncio.apply()
            except Exception:
                pass
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(fn(*args, **kwargs))
    result = fn(*args, **kwargs)
    return result if isinstance(result, str) else str(result)


def _pydantic_model_from_json_schema(tool_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Minimal JSON-schema → Pydantic model mapper for {type, properties, required, default, description}."""
    props = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or [])
    fields: Dict[str, Tuple[Any, Any]] = {}

    for key, spec in props.items():
        py_type: Any = _json_type_to_py(spec.get("type"))
        default = ... if key in required else spec.get("default", None)
        desc = spec.get("description")
        if desc is not None:
            fields[key] = (py_type, Field(default, description=desc))
        else:
            fields[key] = (py_type, default)

    model_name = f"{tool_name}_Args"
    model = create_model(model_name, **fields)  # type: ignore
    return model


def _json_type_to_py(t: Optional[str]) -> Any:
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        None: Any,
    }.get(t, Any)


def _model_from_signature(
    fn: Callable[..., Any],
    *,
    use_docstring_info: bool,
    tool_name: str,
) -> Type[BaseModel]:
    """
    Create a Pydantic model from function signature & (optionally) docstring arg descriptions.
    Supports primitives, Optional, Union, List, Dict, TypedDict, BaseModel, dataclass, Annotated.
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    arg_desc: Dict[str, str] = (
        _extract_arg_descriptions_best_effort(inspect.getdoc(fn) or "")
        if use_docstring_info
        else {}
    )

    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "ctx", "context"):
            continue

        ann = hints.get(name, Any)
        ann, field_info = _normalize_annotation(ann)

        default = param.default if param.default is not inspect._empty else ...
        desc = field_info.description or arg_desc.get(name) or None
        if desc:
            default = Field(default, description=desc)

        fields[name] = (ann, default)

    model_name = f"{tool_name}_Args"
    if not fields:
        return create_model(model_name)  # type: ignore

    model = create_model(model_name, **fields)  # type: ignore
    return model


def _normalize_annotation(ann: Any) -> Tuple[Any, Field]:
    """
    Normalize typing annotations to Pydantic-friendly types.
    Returns (type, Field()) where Field may include description from Annotated.
    """
    # Annotated[T, Field(...)] → capture Field metadata
    if get_origin(ann) is Annotated:
        args = get_args(ann)
        typ = args[0] if args else Any
        extras = args[1:]
        field_meta = Field(None)
        for ex in extras:
            if isinstance(ex, Field):
                field_meta = ex
        inner, inner_field = _normalize_annotation(typ)
        if not field_meta.description and inner_field.description:
            field_meta = Field(field_meta.default, description=inner_field.description)
        return inner, field_meta

    # Optional[T] / Union[T, None]
    if get_origin(ann) is Union:
        args = tuple(get_args(ann))
        if len(args) == 2 and type(None) in args:
            other = args[0] if args[1] is type(None) else args[1]
            inner, meta = _normalize_annotation(other)
            return Optional[inner], meta  # type: ignore

    # List[T], Dict[K, V]
    if get_origin(ann) in (list, List):
        inner_type = get_args(ann)[0] if get_args(ann) else Any
        inner, _ = _normalize_annotation(inner_type)
        return List[inner], Field(None)  # type: ignore
    if get_origin(ann) in (dict, Dict):
        k = get_args(ann)[0] if get_args(ann) else str
        v = get_args(ann)[1] if len(get_args(ann)) > 1 else Any
        key_t, _ = _normalize_annotation(k)
        val_t, _ = _normalize_annotation(v)
        return Dict[key_t, val_t], Field(None)  # type: ignore

    # TypedDict → treat as generic dict (keeps things simple)
    if isinstance(ann, type) and _issubclass_safe(ann, TypedDict):
        return Dict[str, Any], Field(None)

    # pydantic BaseModel subclass
    if isinstance(ann, type) and _issubclass_safe(ann, BaseModel):
        return ann, Field(None)

    # dataclass → convert to pydantic model at runtime
    if dataclasses.is_dataclass(ann):
        return _pyd_model_from_dataclass(ann), Field(None)

    return ann, Field(None)


def _issubclass_safe(a: Any, b: Any) -> bool:
    try:
        return issubclass(a, b)
    except Exception:
        return False


def _pyd_model_from_dataclass(dc_type: Any) -> Type[BaseModel]:
    """Convert a dataclass into a Pydantic model at runtime."""
    hints = get_type_hints(dc_type, include_extras=True)
    fields: Dict[str, Tuple[Any, Any]] = {}
    for f in dataclasses.fields(dc_type):
        ann = hints.get(f.name, Any)
        ann, field_info = _normalize_annotation(ann)
        default = f.default if f.default is not dataclasses.MISSING else ...
        if default is ... and f.default_factory is not dataclasses.MISSING:  # type: ignore
            default = f.default_factory()  # type: ignore
        if field_info.description:
            default = Field(default, description=field_info.description)
        fields[f.name] = (ann, default)
    return create_model(f"{dc_type.__name__}Model", **fields)  # type: ignore


def _extract_arg_descriptions_best_effort(doc: str) -> Dict[str, str]:
    """
    Fast, best-effort arg description parser.
    Looks for a simple 'Args:' block with lines like:
      name: description
      name (type): description
    """
    if not doc:
        return {}
    lines = [ln.rstrip() for ln in doc.splitlines()]
    out: Dict[str, str] = {}

    if "Args:" not in lines:
        return out

    try:
        idx = lines.index("Args:")
    except ValueError:
        return out

    for ln in lines[idx + 1:]:
        stripped = ln.strip()
        if not stripped:
            break
        # lines like "- foo: description" or "foo (type): description"
        raw = stripped.lstrip("- ").strip()
        if ":" not in raw:
            break
        k, v = raw.split(":", 1)
        k = k.split(" ", 1)[0]  # strip "(type)" if present
        out[k.strip()] = v.strip()

    return out
