# abzagent/providers/model_types.py
"""
Per-provider Literal types for Agent's `model=` parameter.

Purpose: IDE autocomplete for known model IDs, without losing the ability to
pass any string (a new model release, a custom deployment, etc.). This is a
pure typing aid — none of it is enforced at runtime, and none of it changes
how a model string is resolved (see SDKConfig.detect_provider in config.py).

Each provider owns its own Literal of the exact bare model-ID strings this
SDK's Agent(model=...) actually expects — not gemini_catalog.py's
"models/"-prefixed live-API resource names, which serve a different purpose
(discovering models via a live API call) and use a different string format
that Agent() doesn't expect.

Extensible by design: adding a new provider means adding one more Literal
here and folding it into KnownModel — nothing about Agent's core logic
needs to change.

## Where this list comes from — NOT memory, NOT guesses

Provider model catalogs genuinely change over time (this list previously
included several models — including an entire generation of Gemini models —
that had since been fully retired, plus a Groq model name that had never
actually been verified against Groq's real catalog). `Literal[...]` members
must be static source code for a type checker to see them, so there is no
way to compute this list from a live API call and have an IDE offer it as
autocomplete — true real-time dynamic generation is impossible here.

Instead: every model below was confirmed via a real call to the provider's
live API (`client.models.list()` for Gemini, `client.models.list()` for
Groq) followed by an actual `generate_content`/`chat.completions.create`
call to confirm it executes, not just that it's listed — some listed models
return 404/429 depending on account tier. Non-conversational endpoints
(embeddings, text-to-speech, speech-to-text, image/video generation,
prompt-guard/safety classifiers, robotics, live/audio-only) were excluded —
they use a different API shape than plain chat, so Agent(model=...) has no
use for them. `-preview`/`-exp`-suffixed models were excluded too, since
they're the ones most likely to be renamed or removed on short notice.

This snapshot WILL drift as providers ship new models — that's expected and
unavoidable for a static list. Run `python scripts/update_model_catalog.py`
periodically (needs GEMINI_API_KEY/GROQ_API_KEY) to check the live catalogs
against this file and see what's changed; update the Literals by hand based
on its output, the same way this list was built. Whatever is here or not,
`Union[KnownModel, str]` means any model string still works identically —
autocomplete is a convenience, not a whitelist.
"""
from __future__ import annotations

from typing import Literal, Union

# Verified live against the Gemini API (see module docstring). General-purpose
# text/chat models only. NOTE: "gemini-2.5-pro" and "gemini-2.5-flash-lite"
# were deliberately excluded despite being listed by the API — both
# reproducibly 404 with "no longer available to new users" against a real
# current key, not a quota/tier 429. That message means genuinely blocked for
# new callers, not "real but rate-limited" — see the inclusion criteria above.
GeminiModel = Literal[
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
]

# Verified live against the Groq API (see module docstring). General-purpose
# text/chat models only — excludes Whisper (speech-to-text), Orpheus
# (text-to-speech), and prompt-guard/safeguard (safety classifiers, not chat).
GroqModel = Literal[
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
    "groq/compound-mini",
    "allam-2-7b",
]

# The union of every provider's known models. New providers extend this.
KnownModel = Union[GeminiModel, GroqModel]
