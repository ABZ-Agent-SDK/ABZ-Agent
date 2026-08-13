"""
Example: Using Gemini provider with ABZ Agent SDK
This verifies that existing Gemini functionality still works.
"""
from abzagent import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Create an agent with a Gemini model
# The SDK will automatically detect it's a Gemini model and use GeminiProvider
agent = Agent(
    name="GeminiAgent",
    instructions="You are a helpful AI assistant powered by Gemini.",
    model="gemini-2.5-flash",  # Gemini model - SDK auto-detects provider
    api_key=os.getenv("GEMINI_API_KEY")  # or set in .env file
)

# Run the agent
response = agent.run("What is 2+2?")
print(response.content)
