"""
Tests for the model-typing DX improvement (abzagent/providers/model_types.py):
IDE autocomplete for Agent(model=...) via per-provider Literal types, while
any string must still work at runtime — Agent(model=...) is fundamentally a
plain string parameter, unenforced at runtime, exactly as before.
"""
import inspect
import os
from typing import Union, get_args, get_type_hints

import pytest

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

from abzagent import Agent, GeminiModel, GroqModel, KnownModel
from abzagent.core import GeminiModel as CoreGeminiModel
from abzagent.core import GroqModel as CoreGroqModel
from abzagent.core import KnownModel as CoreKnownModel
from abzagent.providers.groq_catalog import list_groq_models, best_default, validate_or_suggest
from abzagent.config import SDKConfig


class TestTypesAreExported:
    def test_importable_from_top_level_package(self):
        assert GeminiModel is not None
        assert GroqModel is not None
        assert KnownModel is not None

    def test_importable_from_core_and_identical_to_top_level(self):
        # Same objects both places — one definition, not two competing ones.
        assert CoreGeminiModel is GeminiModel
        assert CoreGroqModel is GroqModel
        assert CoreKnownModel is KnownModel

    def test_in_all_lists(self):
        import abzagent
        import abzagent.core as core
        for name in ("GeminiModel", "GroqModel", "KnownModel"):
            assert name in abzagent.__all__
            assert name in core.__all__


class TestKnownModelContents:
    def test_groq_model_contains_documented_models(self):
        members = get_args(GroqModel)
        for expected in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"):
            assert expected in members

    def test_gemini_model_contains_the_agents_own_default(self):
        """Self-consistency guard: Agent's own default model string must be
        a recognized member of GeminiModel, or the SDK's own default
        wouldn't get autocomplete-recognized as a "known" model."""
        default = inspect.signature(Agent.__init__).parameters["model"].default
        assert default == "gemini-2.0-flash"
        assert default in get_args(GeminiModel)

    def test_gemini_model_uses_bare_names_not_models_prefixed(self):
        """GeminiModel must match what Agent(model=...) actually expects
        (bare names) — not gemini_catalog.py's "models/"-prefixed live-API
        resource names, which are a different, unrelated format."""
        for name in get_args(GeminiModel):
            assert not name.startswith("models/"), f"{name!r} should be a bare model id"

    def test_known_model_is_the_union_of_both_providers(self):
        # Union[Literal[...], Literal[...]] does not auto-flatten into a
        # single combined Literal — get_args(KnownModel) returns the two
        # Literal objects themselves, so they must be unwrapped recursively
        # (matching _contains_known_model's approach) rather than compared
        # as a flat set of strings directly.
        gemini_members = set(get_args(GeminiModel))
        groq_members = set(get_args(GroqModel))

        known_members = set()
        for member in get_args(KnownModel):
            known_members.update(get_args(member))

        assert known_members == gemini_members | groq_members


class TestAgentModelParamAnnotation:
    def test_model_annotation_still_accepts_plain_str(self):
        hints = get_type_hints(Agent.__init__)
        model_hint = hints["model"]
        flattened = get_args(model_hint)
        assert str in flattened  # Optional[Union[KnownModel, str]] -> str must survive flattening

    def test_model_annotation_includes_every_known_model_literal(self):
        """Every GeminiModel/GroqModel string must be reachable from the
        model parameter's resolved annotation via KnownModel — proves the
        Literal types actually made it into Agent's signature, not just
        into model_types.py in isolation."""
        hints = get_type_hints(Agent.__init__)
        assert _contains_known_model(hints["model"])


def _contains_known_model(annotation) -> bool:
    """KnownModel is itself a Union alias, so once combined with Optional[...]
    typing may flatten everything into one big Union rather than nesting
    KnownModel as a distinct member. Check by literal-value membership
    instead of identity, which is robust to how far Python flattens it."""
    seen_literals = set()

    def _walk(ann):
        args = get_args(ann)
        if not args:
            return
        for a in args:
            if isinstance(a, str):
                seen_literals.add(a)
            _walk(a)

    _walk(annotation)
    known = set(get_args(GeminiModel)) | set(get_args(GroqModel))
    return known.issubset(seen_literals)


class TestBackwardCompatibility:
    """The whole point: none of this is enforced at runtime. Any string
    must still work exactly as before."""

    def test_agent_still_accepts_an_unlisted_custom_model_string(self):
        agent = Agent(name="Bot", instructions="Be helpful.", model="my-custom-fine-tuned-model-v7")
        assert agent.model == "my-custom-fine-tuned-model-v7"

    def test_agent_still_accepts_a_known_groq_literal_and_routes_correctly(self):
        agent = Agent(name="Bot", instructions="Be helpful.", model="llama-3.3-70b-versatile")
        assert agent.model == "llama-3.3-70b-versatile"
        assert SDKConfig.detect_provider(agent.model) == "groq"

    def test_agent_default_model_unchanged(self):
        agent = Agent(name="Bot", instructions="Be helpful.")
        assert agent.model == "gemini-2.0-flash"
        assert SDKConfig.detect_provider(agent.model) == "gemini"


class TestGroqCatalogRefactorRegression:
    """groq_catalog._GROQ_MODELS is now derived from GroqModel via
    get_args() instead of being a separately hand-maintained list — these
    guard that the refactor didn't change any existing catalog behavior."""

    def test_list_groq_models_matches_groq_model_literal(self):
        assert list_groq_models() == list(get_args(GroqModel))

    def test_order_preserved_first_entry_unchanged(self):
        assert list_groq_models()[0] == "qwen/qwen3-32b"

    def test_best_default_unaffected(self):
        assert best_default("speed") == "llama-3.1-8b-instant"
        assert best_default("quality") == "llama-3.3-70b-versatile"

    def test_validate_or_suggest_unaffected(self):
        is_valid, suggestion, available = validate_or_suggest("llama-3.3-70b-versatile")
        assert is_valid is True
        assert suggestion is None
        assert "llama-3.3-70b-versatile" in available
