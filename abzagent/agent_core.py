from abzagent import Agent, Memory
from extensions.tools.sdk_knowledge import SDKKnowledgeTool
from tools.file_ops import handle_file_command
from tools.executor import run_command
import os
from dotenv import load_dotenv

load_dotenv()  # Load GEMINI_API_KEY from .env or environment variables

class SuperDevAgent:
    def __init__(self):
        self.memory = Memory()
        self.sdk_tool = SDKKnowledgeTool(docs_path="./abzagent/docs")  # your SDK docs
        self.agent = Agent(
            name="SuperDevAgent",
            instructions=(
                "You are a terminal AI agent. "
                "You know ABZ Agent SDK. "
                "Answer questions about the SDK using SDKKnowledgeTool. "
                "You can also edit files and run commands."
            ),
            model="gemini-2.5-flash",
            memory=self.memory,
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def process_input(self, user_input: str) -> str:
        if "sdk" in user_input.lower() or "abz agent sdk" in user_input.lower():
            return self.sdk_tool.query(user_input)

        response = self.agent.run(user_input)
        text = response.content.lower()

        if any(word in text for word in ["update", "edit", "file", "fix", "create"]):
            return handle_file_command(user_input)
        elif any(word in text for word in ["build", "run", "execute", "test"]):
            return run_command(user_input)
        
        return response.content
