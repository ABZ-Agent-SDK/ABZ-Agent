# abzagent/providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.output import AgentOutputSchema


class ModelProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        output_schema: Optional["AgentOutputSchema"] = None,
        strict: bool = True,
    ) -> str:
        """
        Generate text from the model.

        `output_schema`, when given, means the Agent expects the final answer
        to be JSON matching that schema — providers that support a native
        structured-output mode should use it. `strict=False` means tools are
        active this turn, so providers must not lock generation to the exact
        output schema (a tool-call JSON blob may legitimately be returned
        instead); a looser "valid JSON" mode should be used if available.
        """
        ...
