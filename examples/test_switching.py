"""
Example: Switching between providers
This demonstrates how easy it is to switch between Groq and Gemini models.
"""
from abzagent import Agent
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Testing Groq Provider ===")
groq_agent = Agent(
    name="GroqAgent",
    instructions="You are a helpful assistant.",
    model="llama-3.1-8b-instant",  # Groq model
    api_key=os.getenv("GROQ_API_KEY")
)
groq_response = groq_agent.run("Say hello in one sentence.")
print(f"Groq Response: {groq_response.content}\n")

print("=== Testing Gemini Provider ===")
gemini_agent = Agent(
    name="GeminiAgent",
    instructions="You are a helpful assistant.",
    model="gemini-2.0-flash",  # Gemini model
    api_key=os.getenv("GEMINI_API_KEY")
)
gemini_response = gemini_agent.run("Say hello in one sentence.")
print(f"Gemini Response: {gemini_response.content}")

print("\n✅ Both providers working successfully!")
