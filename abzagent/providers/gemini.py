import os as _os

# extra guard in case this file is imported directly
_os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
_os.environ.setdefault("GRPC_LOG_SEVERITY_OVERRIDE", "ERROR")
_os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from google import genai
from google.genai import types as genai_types
from typing import Optional, TYPE_CHECKING

from .base import ModelProvider
from ..config import SDKConfig

if TYPE_CHECKING:
    from ..core.output import AgentOutputSchema


class GeminiProvider(ModelProvider):
    """
    Wrapper around google-genai (new SDK). Requires a user-supplied API key.
    """

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
        output_schema: Optional["AgentOutputSchema"] = None,
        strict: bool = True,
    ) -> str:
        config = None
        if output_schema is not None and not output_schema.is_plain_text:
            # Gemini natively supports schema-locked JSON via response_json_schema.
            # Only lock the exact shape when no tools are active this turn (strict) —
            # otherwise the model still needs the freedom to emit a tool-call blob.
            config_kwargs = {"response_mime_type": "application/json"}
            if strict:
                config_kwargs["response_json_schema"] = output_schema.json_schema()
            config = genai_types.GenerateContentConfig(**config_kwargs)

        resp = self.client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )
        return getattr(resp, "text", "") or ""
