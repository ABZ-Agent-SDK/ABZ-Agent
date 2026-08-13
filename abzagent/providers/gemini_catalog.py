from __future__ import annotations
from typing import List, Dict, Optional, Tuple, get_args

from .model_types import GeminiModel

try:
    from google import genai
except Exception:
    genai = None  # allow import without SDK installed

# ---- Static fallback list (used if a live API call isn't available) ----
# Single source of truth is the GeminiModel Literal in model_types.py (see
# its docstring for how it's verified against the live Gemini API) — derived
# here by adding the "models/" resource-name prefix this module's live-API
# discovery path uses, rather than maintained as a separate hand-written list
# that could silently drift out of sync.
_FALLBACK_NAMES: List[str] = [f"models/{name}" for name in get_args(GeminiModel)]

# ---- Helpers ----

def _is_generation_model(name: str) -> bool:
    """Accept only generateContent-capable Gemini chat/multimodal models."""
    name = name.lower()
    if not name.startswith("models/"):
        return False
    if any(bad in name for bad in ["embedding", "imagen", "aqa", "gecko", "text-bison", "chat-bison"]):
        return False
    return "gemini" in name

def list_gemini_models(api_key: Optional[str] = None, include_experimental: bool = True) -> List[str]:
    """Live list from API, with static fallback if API is unavailable."""
    names: List[str] = []
    try:
        if genai is None:
            raise RuntimeError("google-genai not installed")

        client = genai.Client(api_key=api_key)
        for m in client.models.list():
            nm = getattr(m, "name", "")
            # Filter for generation models
            if nm and _is_generation_model(nm):
                names.append(nm)
    except Exception:
        names = list(_FALLBACK_NAMES)

    # experimental filter
    if not include_experimental:
        names = [n for n in names if "exp" not in n and "experimental" not in n]

    # unique, stable sorted (by version then variant)
    names = sorted(set(names), key=lambda s: (s.split("/")[-1].replace("latest", "zzz")))
    return names

def tag_model(name: str) -> Dict[str, str]:
    """Classify a model for UX: family, size, speed/quality, generation."""
    n = name.lower()
    family = "gemini"
    speed = "balanced"
    quality = "balanced"
    size = "standard"

    if "flash-8b" in n:
        size = "8B"
        speed = "fastest"
    elif "flash" in n:
        speed = "fast"
    if "pro" in n:
        quality = "high"
    if "2.0" in n or "2.5" in n:
        family = "gemini-2.x"
    if "exp" in n or "experimental" in n:
        quality = f"{quality} (exp)"

    return {
        "family": family,
        "speed": speed,
        "quality": quality,
        "size": size,
    }

def best_default(goal: str = "balanced") -> str:
    """
    goal in {"speed","balanced","quality"}
    Returns a sensible default.
    """
    models = list_gemini_models(include_experimental=False)
    # quality: prefer pro
    if goal == "quality":
        for cand in ["models/gemini-pro-latest"]:
            if cand in models:
                return cand
    # speed: prefer the flash-lite tier
    if goal == "speed":
        for cand in ["models/gemini-flash-lite-latest", "models/gemini-3.1-flash-lite",
                     "models/gemini-3.5-flash-lite"]:
            if cand in models:
                return cand
    # balanced: prefer flash
    for cand in ["models/gemini-2.5-flash", "models/gemini-flash-latest"]:
        if cand in models:
            return cand
    # Fallback
    return models[0] if models else "models/gemini-2.5-flash"

def validate_or_suggest(chosen: str, include_experimental: bool = True) -> Tuple[bool, Optional[str], List[str]]:
    """
    Returns (is_valid, suggestion, available).
    suggestion is the closest match by simple heuristic if invalid.
    """
    avail = list_gemini_models(include_experimental=include_experimental)
    if chosen in avail:
        return True, None, avail
    # naive suggestion by postfix distance
    want = chosen.split("/")[-1].lower()
    def score(n: str) -> int:
        post = n.split("/")[-1].lower()
        # small score = closer
        return sum(a != b for a, b in zip(post, want)) + abs(len(post) - len(want))
    if avail:
        suggestion = min(avail, key=score)
        return False, suggestion, avail
    return False, None, avail
