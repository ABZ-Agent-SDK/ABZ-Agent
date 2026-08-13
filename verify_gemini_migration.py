from abzagent import Agent
import os

# Create an agent with a Gemini model
# The SDK will automatically detect it's a Gemini model and use GeminiProvider
agent = Agent(
    name="GeminiAgent",
    instructions="You are a helpful AI assistant powered by Gemini.",
    model="gemini-2.5-flash",  # Gemini model - SDK auto-detects provider
    api_key=os.getenv("GEMINI_API_KEY") 
)

# Run the agent
print("Running Gemini migration test...")
try:
    response = agent.run("What is 2+2?")
    print(f"Response: {response.content}")
    print("✅ Gemini migration verification passed!")
except Exception as e:
    print(f"❌ Gemini migration verification failed: {e}")
