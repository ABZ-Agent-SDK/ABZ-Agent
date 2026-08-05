from dotenv import load_dotenv
load_dotenv()  # reads GROQ_API_KEY from .env or env
import os
from abzagent import Agent, Memory

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def main():
    agent = Agent(
        name='My Agent',
        instructions='Be helpful and concise.',
        model='groq/compound',
        memory=Memory(),
        api_key=GROQ_API_KEY
    )
    
    print("🤖 Agent started. Type 'exit' to quit.")
    while True:
        user_input = input("> ")
        if user_input.lower() in ('exit', 'quit'):
            print("👋 Goodbye!")
            break
        response = agent.run(user_input)
        print("Agent response:", response.content)

if __name__ == "__main__":
    main()
