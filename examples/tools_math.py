from abzagent import function_tool
from abzagent.Tools.tools_math import safe_eval

@function_tool()  # ← parentheses REQUIRED in your SDK
def calc(expression: str) -> str:
    """Evaluate arithmetic like '2+2*10'."""
    try:
        return str(safe_eval(expression))
    except Exception:
        return "Error: only arithmetic expressions are supported."
