# abzagent/providers/groq.py
from __future__ import annotations
import json
import os as _os

from groq import Groq
from typing import List, Optional, TYPE_CHECKING

from .base import ModelProvider, GenerationResult
from ..config import SDKConfig
from ..core.tools import ToolCall, ToolSchema

if TYPE_CHECKING:
    from ..core.output import AgentOutputSchema


class GroqProvider(ModelProvider):
    """
    Wrapper around Groq API. Requires a user-supplied API key.
    """

    supports_native_tools = True

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
        tools: Optional[List[ToolSchema]] = None,
        output_schema: Optional["AgentOutputSchema"] = None,
        strict: bool = True,
    ) -> GenerationResult:
        """
        Generate content using Groq's chat completions API.
        """
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self._model_name,
        }

        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        elif output_schema is not None and not output_schema.is_plain_text:
            # Groq's "json_schema" strict mode is only available on select models,
            # so we lean on the broadly-supported "json_object" JSON mode here and
            # let the schema described in the prompt + Agent-side Pydantic
            # validation (with a repair retry) do the exact-shape enforcement.
            request["response_format"] = {"type": "json_object"}

        try:
            chat_completion = self.client.chat.completions.create(**request)
            message = chat_completion.choices[0].message

            if message.tool_calls:
                calls = []
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    calls.append(ToolCall(tool=tc.function.name, args=args))
                return GenerationResult(tool_calls=calls)

            return GenerationResult(text=message.content or "")
        except Exception as e:
            return GenerationResult(text=f"[Groq Error] {str(e)}")
