# abzagent/providers/groq.py
from __future__ import annotations
import os as _os

from groq import Groq
from typing import Optional, TYPE_CHECKING

from .base import ModelProvider
from ..config import SDKConfig

if TYPE_CHECKING:
    from ..core.output import AgentOutputSchema


class GroqProvider(ModelProvider):
    """
    Wrapper around Groq API. Requires a user-supplied API key.
    """

    def __init__(self, config: SDKConfig):
        self.config = config.require_key()  # enforce key

        # Initialize Groq client
        self.client = Groq(
            api_key=self.config.api_key
        )
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
        """
        Generate content using Groq's chat completions API.
        """
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self._model_name,
        }

        if output_schema is not None and not output_schema.is_plain_text:
            # Groq's "json_schema" strict mode is only available on select models,
            # so we lean on the broadly-supported "json_object" JSON mode here and
            # let the schema described in the prompt + Agent-side Pydantic
            # validation (with a repair retry) do the exact-shape enforcement.
            request["response_format"] = {"type": "json_object"}

        try:
            chat_completion = self.client.chat.completions.create(**request)

            # Extract the response content
            return chat_completion.choices[0].message.content or ""
        except Exception as e:
            return f"[Groq Error] {str(e)}"
