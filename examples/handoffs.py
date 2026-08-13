"""
Handoffs example.

Pass a list of specialist agents as `handoffs=[...]` and the SDK automatically
routes, transfers memory/context, and returns the specialist's own answer as
the final result — no manual routing logic required.
"""

from abzagent import Agent

research_agent = Agent(name="Research", instructions="Research topics thoroughly and report facts.")
writer_agent = Agent(name="Writer", instructions="Write clear, engaging copy from research notes.")
review_agent = Agent(name="Review", instructions="Review and polish drafts for clarity and tone.")

planner = Agent(
    name="Planner",
    instructions="Route tasks to the correct specialist. Don't answer directly yourself.",
    model="gemini-2.5-flash",
    handoffs=[research_agent, writer_agent, review_agent],
)

result = planner.run("Write a short blog post about black holes.")

print("Answered by:", result.last_agent.name)
print(result.content)
