from __future__ import annotations
from typing import List, Dict, Optional, Tuple

try:
    from google import genai
except Exception:
    genai = None  # allow import without SDK installed

# ---- Static fallback list (used if a live API call isn't available) ----
_FALLBACK_NAMES: List[str] = [
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro-002",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-flash-001-tuning",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-002",
    "models/gemini-1.5-flash-8b",
    "models/gemini-1.5-flash-8b-001",
    "models/gemini-1.5-flash-8b-latest",
    "models/gemini-1.5-flash-8b-exp-0827",
    "models/gemini-1.5-flash-8b-exp-0924",
    "models/gemini-2.5-pro-exp-03-25",
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-exp-image-generation",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-preview-02-05",
    "models/gemini-2.0-flash-lite-preview",
    "models/gemini-2.0-pro-exp",
    "models/gemini-2.0-pro-exp-02-05",
    "models/gemini-exp-1206",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    "models/gemini-2.0-flash-thinking-exp",
    "models/gemini-2.0-flash-thinking-exp-1219",
    # keep embeddings/imagen/PaLM out of default picks
]

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
    # quality: prefer pro (1.5-pro-002 then latest), else 2.0 flash if only option
    if goal == "quality":
        for cand in ["models/gemini-1.5-pro-002", "models/gemini-1.5-pro-latest", "models/gemini-1.5-pro"]:
            if cand in models:
                return cand
    # speed: prefer flash-8b or flash 2.0/1.5
    if goal == "speed":
        for cand in ["models/gemini-1.5-flash-8b-001", "models/gemini-1.5-flash-8b-latest",
                     "models/gemini-2.0-flash-001", "models/gemini-1.5-flash-002",
                     "models/gemini-1.5-flash-latest", "models/gemini-1.5-flash"]:
            if cand in models:
                return cand
    # balanced: prefer 1.5-pro-002 then 1.5-flash-002
    for cand in ["models/gemini-1.5-pro-002", "models/gemini-1.5-flash-002",
                 "models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest"]:
        if cand in models:
            return cand
    # Fallback
    return models[0] if models else "models/gemini-1.5-pro"

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
