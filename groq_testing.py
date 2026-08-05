from abzagent import Agent , Memory

# Use Groq (auto-detected)
agent = Agent(
    name="MyAgent",
    instructions="You are helpful.",
    model="qwen/qwen3-32b"  # That's it!,
    memory = Memory()
)

response = agent.run("Hello!")

print(response.content)