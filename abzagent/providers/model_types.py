# abzagent/providers/model_types.py
"""
Per-provider Literal types for Agent's `model=` parameter.

Purpose: IDE autocomplete for known model IDs, without losing the ability to
pass any string (a new model release, a custom deployment, etc.). This is a
pure typing aid — none of it is enforced at runtime, and none of it changes
how a model string is resolved (see SDKConfig.detect_provider in config.py).

Each provider owns its own Literal of the exact bare model-ID strings this
SDK's Agent(model=...) actually expects and documents (see README.md's model
tables) — not gemini_catalog.py's "models/"-prefixed live-API resource
names, which serve a different purpose (discovering models via a live API
call) and use a different string format that Agent() doesn't expect.

Extensible by design: adding a new provider means adding one more Literal
here and folding it into KnownModel — nothing about Agent's core logic
needs to change.
"""
from __future__ import annotations

from typing import Literal, Union

GeminiModel = Literal[
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro-exp-03-25",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

GroqModel = Literal[
    "qwen/qwen3-32b",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "gemma-7b-it",
]

# The union of every provider's known models. New providers extend this.
KnownModel = Union[GeminiModel, GroqModel]
