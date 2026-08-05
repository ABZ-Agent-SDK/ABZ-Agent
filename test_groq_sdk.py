"""
Test Groq provider integration with ABZ Agent SDK
"""
from abzagent import Agent
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Testing Groq Provider with ABZ Agent SDK ===\n")

# Create an agent with a Groq model
agent = Agent(
    name="GroqTestAgent",
    instructions="You are a helpful AI assistant. Be concise.",
    model="qwen/qwen3-32b",  # Groq model - SDK auto-detects provider
    api_key=os.getenv("GROQ_API_KEY")
)

# Test the agent
print("Question: Explain the importance of fast language models in 2-3 sentences.\n")
response = agent.run("Explain the importance of fast language models in 2-3 sentences.")
print(f"Response:\n{response.content}\n")
print("[SUCCESS] Groq provider working successfully!")
