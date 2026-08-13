"""
Example: Using Groq provider with ABZ Agent SDK
This demonstrates how to use Groq models (like Qwen) with the SDK.
"""
from abzagent import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Create an agent with a Groq model
# The SDK will automatically detect it's a Groq model and use GroqProvider
agent = Agent(
    name="GroqAgent",
    instructions="You are a helpful AI assistant powered by Groq.",
    model="qwen/qwen3.6-27b",  # Groq model - SDK auto-detects provider
    api_key=os.getenv("GROQ_API_KEY")  # or set in .env file
)

# Run the agent
response = agent.run("Explain the importance of fast language models in 2-3 sentences.")
print(response.content)
