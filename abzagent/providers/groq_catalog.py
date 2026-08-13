# abzagent/providers/groq_catalog.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, get_args

from .model_types import GroqModel

try:
    from groq import Groq
except Exception:
    Groq = None  # allow import without the groq package installed

# ---- Static fallback list (used if a live API call isn't available) ----
# Single source of truth is the GroqModel Literal in model_types.py (see its
# docstring for how it's verified against Groq's live catalog) — derived
# here via get_args() rather than duplicated, so the two can't drift apart.
_FALLBACK_MODELS: List[str] = list(get_args(GroqModel))

# Non-chat endpoints Groq's /models list also returns (speech-to-text,
# text-to-speech, safety classifiers) — filtered out of live results the
# same way they're already excluded from GroqModel, since Agent(model=...)
# has no use for them (different API shape than chat.completions).
_NON_CHAT_MARKERS = ("whisper", "orpheus", "prompt-guard", "safeguard")


def list_groq_models(api_key: Optional[str] = None) -> List[str]:
    """Live list from Groq's API, with static fallback if unavailable."""
    if Groq is not None:
        try:
            client = Groq(api_key=api_key) if api_key else Groq()
            models = client.models.list()
            names = [
                m.id for m in models.data
                if getattr(m, "active", True) and not any(marker in m.id.lower() for marker in _NON_CHAT_MARKERS)
            ]
            if names:
                return sorted(names)
        except Exception:
            pass
    return list(_FALLBACK_MODELS)


def tag_model(name: str) -> Dict[str, str]:
    """Classify a Groq model for UX: family, size, speed/quality."""
    n = name.lower()
    family = "unknown"
    speed = "balanced"
    quality = "balanced"
    size = "standard"

    # Detect family
    if "qwen" in n:
        family = "qwen"
    elif "llama" in n:
        family = "llama"
    elif "openai" in n or "gpt-oss" in n:
        family = "gpt-oss"
    elif "groq/compound" in n:
        family = "compound"
    elif "allam" in n:
        family = "allam"
    elif "mixtral" in n:
        family = "mixtral"
    elif "deepseek" in n:
        family = "deepseek"
    elif "gemma" in n:
        family = "gemma"

    # Detect size and speed
    if "120b" in n or "70b" in n or "72b" in n:
        size = "70B+"
        quality = "high"
        speed = "slower"
    elif "27b" in n or "32b" in n:
        size = "27-32B"
        quality = "high"
        speed = "balanced"
    elif "20b" in n:
        size = "20B"
        speed = "balanced"
    elif "8b" in n or "7b" in n:
        size = "7-8B"
        speed = "fast"
    if "instant" in n:
        speed = "fastest"
    if "mini" in n:
        speed = "fast"

    return {
        "family": family,
        "speed": speed,
        "quality": quality,
        "size": size,
    }


def best_default(goal: str = "balanced") -> str:
    """
    goal in {"speed","balanced","quality"}
    Returns a sensible default Groq model.
    """
    models = list_groq_models()

    # quality: prefer larger models
    if goal == "quality":
        for cand in ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]:
            if cand in models:
                return cand

    # speed: prefer smaller/instant models
    if goal == "speed":
        for cand in ["llama-3.1-8b-instant", "allam-2-7b"]:
            if cand in models:
                return cand

    # balanced: prefer mid-size models
    for cand in ["qwen/qwen3.6-27b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]:
        if cand in models:
            return cand

    # Fallback
    return models[0] if models else "llama-3.1-8b-instant"


def validate_or_suggest(chosen: str) -> Tuple[bool, Optional[str], List[str]]:
    """
    Returns (is_valid, suggestion, available).
    suggestion is the closest match if invalid.
    """
    avail = list_groq_models()
    if chosen in avail:
        return True, None, avail

    # Simple suggestion by substring matching
    want = chosen.lower()
    matches = [m for m in avail if want in m.lower() or m.lower() in want]

    if matches:
        return False, matches[0], avail

    # Fallback to default
    return False, best_default(), avail
