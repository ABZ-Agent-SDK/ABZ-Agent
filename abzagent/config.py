# abzagent/config.py
from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SDKConfig:
    """
    Central runtime config for ABZ Agent SDK (Gemini & Groq).
NOTE: The API key must be provided by the USER (never hardcode).

The SDK automatically loads environment variables from a `.env` file, so you don't need to manually verify it.
    """
    model: str = os.getenv("ABZ_MODEL", "gemini-2.5-flash")
    api_key: str = ""
    provider: str = "gemini"  # "gemini" or "groq"
    temperature: float = float(os.getenv("ABZ_TEMPERATURE", "0.4"))
    max_iterations: int = int(os.getenv("ABZ_MAX_ITERS", "4"))
    verbose: bool = os.getenv("ABZ_VERBOSE", "1") == "1"

    def __post_init__(self):
        """Auto-detect provider from model name if not explicitly set."""
        if not self.api_key:
            # Try to get API key based on provider
            if self.provider == "groq":
                self.api_key = os.getenv("GROQ_API_KEY", "")
            else:
                self.api_key = os.getenv("GEMINI_API_KEY", "")

    @staticmethod
    def detect_provider(model: str) -> str:
        """
        Auto-detect provider based on model name.
        Returns "groq" or "gemini".

        Checks the known-current Groq model list first (exact match, kept in
        sync with GroqModel in providers/model_types.py — see that module's
        docstring for how it's verified against Groq's live catalog), then
        falls back to substring patterns for anything not in that list, so a
        custom/future/unlisted Groq model with a recognizable family name
        still routes correctly. Deferred import to avoid a circular import
        with providers/__init__.py, which this module is imported from.
        """
        from .providers.model_types import GroqModel
        from typing import get_args

        if model in get_args(GroqModel):
            return "groq"

        model_lower = model.lower()

        # Groq model family patterns (fallback for models not in the known list)
        groq_patterns = ["qwen/", "llama", "mixtral", "deepseek", "gemma2", "gemma-"]
        if any(pattern in model_lower for pattern in groq_patterns):
            return "groq"

        # Default to Gemini for gemini models or unknown
        return "gemini"

    def has_key(self) -> bool:
        return bool(self.api_key)

    def require_key(self) -> "SDKConfig":
        """
        Hard-fail unless the USER provided a key.
        Ways to provide:
          - export GEMINI_API_KEY or GROQ_API_KEY in env (or .env via python-dotenv)
          - pass Agent(api_key="...") which overrides this field
        """
        if not self.api_key:
            key_name = "GROQ_API_KEY" if self.provider == "groq" else "GEMINI_API_KEY"
            raise RuntimeError(
                f"{key_name} is not set. Provide YOUR OWN API key - "
                f"(environment variable or .env) or pass Agent(api_key='...')."
            )
        return self
