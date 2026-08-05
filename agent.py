from abzagent import Agent


agent = Agent(
    name = "Assistant",
    instructions="You are a helpful assistant.",
    model="qwen/qwen3-32b"
)

result = agent.run("Hello, how are you?")
print(result.content)