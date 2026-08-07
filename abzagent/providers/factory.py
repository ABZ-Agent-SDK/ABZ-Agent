# abzagent/providers/factory.py
from __future__ import annotations

import os
from typing import Optional

from ..config import SDKConfig
from .base import ModelProvider
from .gemini import GeminiProvider
from .groq import GroqProvider


def resolve_provider(model: str, *, api_key: Optional[str] = None) -> ModelProvider:
    """
    Build a standalone ModelProvider for `model`, resolving the API key from
    `api_key` or the appropriate env var (GROQ_API_KEY/GEMINI_API_KEY) if not
    given. Mirrors Agent.__init__'s own provider-construction logic — kept as
    a separate reusable helper since core.guardrails cannot import from
    core.agent (would create a circular import) but can safely import from
    providers/.
    """
    provider_type = SDKConfig.detect_provider(model)

    resolved_key = api_key
    if not resolved_key:
        env_var = "GROQ_API_KEY" if provider_type == "groq" else "GEMINI_API_KEY"
        resolved_key = os.getenv(env_var, "")

    if not resolved_key:
        key_name = "GROQ_API_KEY" if provider_type == "groq" else "GEMINI_API_KEY"
        raise RuntimeError(f"{key_name} missing — set in env or .env")

    cfg = SDKConfig(model=model, api_key=resolved_key, provider=provider_type)
    return GroqProvider(cfg) if provider_type == "groq" else GeminiProvider(cfg)
