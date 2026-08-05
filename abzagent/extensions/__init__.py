try:
    from .handoffs_util import parse_handoff_output
except Exception:
    def parse_handoff_output(text: str):  # fallback
        return (None, text)

__all__ = ["parse_handoff_output"]
