# abzagent/providers/groq_catalog.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

# ---- Static list of popular Groq models ----
_GROQ_MODELS: List[str] = [
    # Qwen models
    "qwen/qwen3-32b",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    
    # Llama models
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    
    # Mixtral models
    "mixtral-8x7b-32768",
    
    # DeepSeek models
    "deepseek-r1-distill-llama-70b",
    
    # Gemma models
    "gemma2-9b-it",
    "gemma-7b-it",
]

def list_groq_models() -> List[str]:
    """Return list of available Groq models."""
    return list(_GROQ_MODELS)

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
    elif "mixtral" in n:
        family = "mixtral"
    elif "deepseek" in n:
        family = "deepseek"
    elif "gemma" in n:
        family = "gemma"
    
    # Detect size and speed
    if "70b" in n or "72b" in n:
        size = "70B+"
        quality = "high"
        speed = "slower"
    elif "32b" in n:
        size = "32B"
        quality = "high"
        speed = "balanced"
    elif "8b" in n or "7b" in n:
        size = "7-8B"
        speed = "fast"
    elif "instant" in n:
        speed = "fastest"
    
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
        for cand in ["llama-3.3-70b-versatile", "qwen/qwen-2.5-72b-instruct", "llama-3.1-70b-versatile"]:
            if cand in models:
                return cand
    
    # speed: prefer smaller/instant models
    if goal == "speed":
        for cand in ["llama-3.1-8b-instant", "llama3-8b-8192", "qwen/qwen-2.5-7b-instruct"]:
            if cand in models:
                return cand
    
    # balanced: prefer mid-size models
    for cand in ["qwen/qwen3-32b", "qwen/qwen-2.5-32b-instruct", "llama-3.1-70b-versatile"]:
        if cand in models:
            return cand
    
    # Fallback
    return models[0] if models else "qwen/qwen3-32b"

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
