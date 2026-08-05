# abzagent/providers/__init__.py
from __future__ import annotations
import os

# Silence gRPC / Gemini warnings globally
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_SUPPRESS_LOGS"] = "1"
from .base import ModelProvider
from .gemini import GeminiProvider
from .groq import GroqProvider

# Optional Gemini catalog (don't hard fail if missing in minimal installs)
try:
    from .gemini_catalog import (
        list_gemini_models,
        best_default as gemini_best_default,
        validate_or_suggest as gemini_validate_or_suggest,
        tag_model as gemini_tag_model,
    )
except Exception:  # pragma: no cover
    list_gemini_models = None  # type: ignore
    gemini_best_default = None        # type: ignore
    gemini_validate_or_suggest = None # type: ignore
    gemini_tag_model = None           # type: ignore

# Optional Groq catalog
try:
    from .groq_catalog import (
        list_groq_models,
        best_default as groq_best_default,
        validate_or_suggest as groq_validate_or_suggest,
        tag_model as groq_tag_model,
    )
except Exception:  # pragma: no cover
    list_groq_models = None  # type: ignore
    groq_best_default = None        # type: ignore
    groq_validate_or_suggest = None # type: ignore
    groq_tag_model = None           # type: ignore

__all__ = [
    "ModelProvider",
    "GeminiProvider",
    "GroqProvider",
    "list_gemini_models",
    "gemini_best_default",
    "gemini_validate_or_suggest",
    "gemini_tag_model",
    "list_groq_models",
    "groq_best_default",
    "groq_validate_or_suggest",
    "groq_tag_model",
]
