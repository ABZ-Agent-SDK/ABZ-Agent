# abzagent/core/output.py
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import get_origin, is_typeddict

WRAPPER_KEY = "value"

_JSON_BLOB_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


class ModelBehaviorError(Exception):
    """
    Raised when a model's raw output could not be parsed/validated into the
    Agent's declared `output_type`. Distinct from a Pydantic ValidationError:
    this signals a bad *model response*, not a bad call from your code.
    """


def _needs_wrapping(output_type: Any) -> bool:
    """
    JSON Schema (and every provider's structured-output mode) requires the
    top-level schema to describe a JSON object. Pydantic models, dataclasses,
    TypedDicts, and dict-likes already do. Everything else (str/int/list/...)
    gets wrapped under a synthetic {"value": ...} envelope.
    """
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return False
    if dataclasses.is_dataclass(output_type):
        return False
    if is_typeddict(output_type):
        return False
    if get_origin(output_type) is dict:
        return False
    return True


class AgentOutputSchema:
    """
    Wraps an Agent's `output_type`. Owns everything structured-output related:
    generating a JSON Schema, describing it to the model in the prompt, and
    validating the model's raw text back into a real Python object.

    Providers may additionally use `.json_schema()` to natively constrain
    generation when they support it; the prompt instructions + validation
    here are the provider-agnostic fallback that always applies.
    """

    def __init__(self, output_type: Type[Any], *, max_retries: int = 1):
        self.output_type = output_type
        self.max_retries = max_retries
        self._adapter: TypeAdapter = TypeAdapter(output_type)
        self._wrapped = _needs_wrapping(output_type)
        self._schema = self._build_schema()

    @property
    def is_plain_text(self) -> bool:
        return self.output_type in (None, str)

    @property
    def name(self) -> str:
        raw = getattr(self.output_type, "__name__", None) or str(self.output_type)
        safe = _SAFE_NAME_RE.sub("_", raw).strip("_") or "Output"
        return safe[:64]

    def json_schema(self) -> Dict[str, Any]:
        return self._schema

    def prompt_instructions(self) -> str:
        schema_str = json.dumps(self._schema, indent=2)
        shape = "a single JSON object" if self._wrapped or self._is_object_schema() else "JSON"
        return (
            f"You must respond with ONLY {shape} matching the schema below named "
            f"'{self.name}'. No markdown fences, no commentary, no extra text — JSON only.\n"
            f"JSON Schema:\n{schema_str}"
        )

    def validate_json(self, text: str) -> Any:
        blob = _extract_json_blob(text)
        if blob is None:
            raise ModelBehaviorError(
                f"Expected JSON output matching '{self.name}', but no JSON was found "
                f"in the model's response: {text!r}"
            )
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as e:
            raise ModelBehaviorError(f"Model output was not valid JSON: {e}") from e

        if self._wrapped and isinstance(data, dict) and WRAPPER_KEY in data:
            data = data[WRAPPER_KEY]

        try:
            return self._adapter.validate_python(data)
        except ValidationError as e:
            raise ModelBehaviorError(
                f"Model output did not match the '{self.name}' schema: {e}"
            ) from e

    def _is_object_schema(self) -> bool:
        return self._schema.get("type") == "object"

    def _build_schema(self) -> Dict[str, Any]:
        raw = self._adapter.json_schema()
        if not self._wrapped:
            return raw

        defs = raw.pop("$defs", None)
        wrapped: Dict[str, Any] = {
            "type": "object",
            "properties": {WRAPPER_KEY: raw},
            "required": [WRAPPER_KEY],
        }
        if defs:
            wrapped["$defs"] = defs
        return wrapped


def _extract_json_blob(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    m = _JSON_BLOB_RE.search(text)
    return m.group(1) if m else None
