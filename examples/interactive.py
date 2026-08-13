"""
Interactive mode example.

No manual while-loop, no input() handling, no exit-command parsing —
run(interactive=True) handles all of it for you. Same method as a
single request (agent.run("...")), just a different mode.
"""

from abzagent import Agent

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gemini-2.5-flash",
)

agent.run(interactive=True)
