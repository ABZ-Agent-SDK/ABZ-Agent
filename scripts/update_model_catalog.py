#!/usr/bin/env python
"""
Check the live Gemini/Groq model catalogs against abzagent/providers/model_types.py
and report drift. This is the documented update mechanism referenced in that
file's docstring.

`Literal[...]` members must be static source text for type checkers to see
them as autocomplete — there is no way to regenerate GeminiModel/GroqModel
automatically and have IDEs pick it up. This script instead does the next
best thing: it calls each provider's real /models endpoint, diffs the result
against the current Literal, and tells a human exactly what to look at.

It does NOT edit model_types.py automatically, and it does NOT add anything
to the Literal on your behalf. Membership in a provider's list is necessary
but not sufficient for inclusion (see model_types.py's docstring) — some
listed models are non-chat endpoints (embeddings, TTS/STT, safety
classifiers, image/video generation), tier-gated, or about to be retired.
A model reported here as "new" still needs a real chat/generate_content call
to confirm it actually works before it belongs in the Literal.

Usage:
    python scripts/update_model_catalog.py

Needs GEMINI_API_KEY and/or GROQ_API_KEY in the environment or a .env file.
Either key may be omitted; that provider's section is skipped with a note.
"""
from __future__ import annotations

import os
import sys
from typing import List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # stdout doesn't support reconfigure (unusual, but non-fatal)

from dotenv import load_dotenv

load_dotenv()

from abzagent.providers.model_types import GeminiModel, GroqModel
from typing import get_args

# Endpoint/capability markers that mean "not a plain chat model" — mirrors
# the exclusions already documented in model_types.py, so this script's
# notion of "irrelevant to Agent(model=...)" stays consistent with the
# Literal's own stated inclusion criteria.
_GEMINI_NON_CHAT_MARKERS = (
    "embedding", "imagen", "aqa", "gecko", "text-bison", "chat-bison",
    "tts", "image", "audio", "veo", "learnlm", "gemma",
)
_GROQ_NON_CHAT_MARKERS = ("whisper", "orpheus", "prompt-guard", "safeguard")


def check_gemini() -> None:
    print("=" * 70)
    print("GEMINI")
    print("=" * 70)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set — skipping live check.\n")
        return

    try:
        from google import genai
    except Exception as e:
        print(f"google-genai not installed ({e}) — skipping live check.\n")
        return

    known: Set[str] = set(get_args(GeminiModel))

    try:
        client = genai.Client(api_key=api_key)
        live: List[str] = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            if not name.startswith("models/"):
                continue
            bare = name.split("/", 1)[1]
            if "gemini" not in bare.lower():
                continue
            if any(marker in bare.lower() for marker in _GEMINI_NON_CHAT_MARKERS):
                continue
            if "-preview" in bare.lower() or "-exp" in bare.lower():
                continue
            live.append(bare)
    except Exception as e:
        print(f"Live API call failed ({e}) — skipping live check.\n")
        return

    _report(known, set(live))


def check_groq() -> None:
    print("=" * 70)
    print("GROQ")
    print("=" * 70)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set — skipping live check.\n")
        return

    try:
        from groq import Groq
    except Exception as e:
        print(f"groq package not installed ({e}) — skipping live check.\n")
        return

    known: Set[str] = set(get_args(GroqModel))

    try:
        client = Groq(api_key=api_key)
        live = [
            m.id for m in client.models.list().data
            if getattr(m, "active", True)
            and not any(marker in m.id.lower() for marker in _GROQ_NON_CHAT_MARKERS)
        ]
    except Exception as e:
        print(f"Live API call failed ({e}) — skipping live check.\n")
        return

    _report(known, set(live))


def _report(known: Set[str], live: Set[str]) -> None:
    new = sorted(live - known)
    missing = sorted(known - live)

    if new:
        print(f"\nIn the live catalog but NOT in the Literal ({len(new)}):")
        for name in new:
            print(f"  + {name}")
        print(
            "  Before adding any of these: confirm with a real chat/generate_content\n"
            "  call that it actually works (not just listed — see model_types.py)."
        )
    else:
        print("\nNo new models found in the live catalog.")

    if missing:
        print(f"\nIn the Literal but NOT in the live catalog ({len(missing)}):")
        for name in missing:
            print(f"  - {name}")
        print(
            "  These may be retired, renamed, or temporarily unlisted. Verify with\n"
            "  a real call before removing — a 404 without a listing is not proof\n"
            "  by itself, but a 404 saying the model 'is no longer available' is."
        )
    else:
        print("\nEvery Literal entry is still present in the live catalog.")

    print()


if __name__ == "__main__":
    check_gemini()
    check_groq()
    print(
        "Reminder: this script only reports drift. Update GeminiModel/GroqModel\n"
        "in abzagent/providers/model_types.py by hand, the same way this list was\n"
        "originally built — verify each candidate against a real generate call,\n"
        "not just its presence in the list above."
    )
