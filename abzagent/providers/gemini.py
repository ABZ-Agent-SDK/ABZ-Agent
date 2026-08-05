import os as _os

# extra guard in case this file is imported directly
_os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
_os.environ.setdefault("GRPC_LOG_SEVERITY_OVERRIDE", "ERROR")
_os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from google import genai
from google.genai import types as genai_types
from typing import List, Optional, TYPE_CHECKING

from .base import ModelProvider, GenerationResult
from ..config import SDKConfig
from ..core.tools import ToolCall, ToolSchema

if TYPE_CHECKING:
    from ..core.output import AgentOutputSchema


class GeminiProvider(ModelProvider):
    """
    Wrapper around google-genai (new SDK). Requires a user-supplied API key.
    """

    supports_native_tools = True

    def __init__(self, config: SDKConfig):
        self.config = config.require_key()  # enforce key
        self.client = genai.Client(api_key=self.config.api_key)
        self._model_name = self.config.model

    @property
    def model(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        *,
        tools: Optional[List[ToolSchema]] = None,
        output_schema: Optional["AgentOutputSchema"] = None,
        strict: bool = True,
    ) -> GenerationResult:
        config_kwargs = {}

        if tools:
            declarations = [
                genai_types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters,
                )
                for t in tools
            ]
            config_kwargs["tools"] = [genai_types.Tool(function_declarations=declarations)]
            # We pass structured tool schemas, not raw Python callables, so
            # google-genai's automatic function calling shouldn't engage — disable
            # it explicitly anyway as cheap insurance against SDK internals.
            config_kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(
                disable=True
            )
        elif output_schema is not None and not output_schema.is_plain_text:
            # Gemini natively supports schema-locked JSON via response_json_schema.
            # Only lock the exact shape when no tools are active this turn (strict) —
            # otherwise the model still needs the freedom to make a tool call.
            config_kwargs["response_mime_type"] = "application/json"
            if strict:
                config_kwargs["response_json_schema"] = output_schema.json_schema()

        config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        resp = self.client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )

        if resp.function_calls:
            return GenerationResult(
                tool_calls=[
                    ToolCall(tool=fc.name, args=fc.args or {}) for fc in resp.function_calls
                ]
            )
        return GenerationResult(text=getattr(resp, "text", "") or "")
