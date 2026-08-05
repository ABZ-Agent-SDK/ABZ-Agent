# abzagent/extensions/handoff_prompt.py
RECOMMENDED_PROMPT_PREFIX = """\
You can DELEGATE tasks to other specialized agents using tools named like:
- transfer_to_<agent_name_slug>

When you decide another agent is better suited, call the correct transfer tool ONCE with any minimal context it needs. 
Return ONLY a JSON object for the tool call, e.g.:

{"tool":"transfer_to_refund_agent","args":{"message":"Customer is asking for a refund for order #123."}}

After delegating, do not repeat the answer yourself.
"""

def prompt_with_handoff_instructions(prompt: str) -> str:
    return f"{RECOMMENDED_PROMPT_PREFIX}\n{prompt}"
