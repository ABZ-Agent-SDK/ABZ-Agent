from dotenv import load_dotenv
from abzagent.core.agent import Agent
from abzagent.core.handoffs import handoff
import os

load_dotenv()

medical_agent = Agent(name="MedicalAssistant", instructions="Handle medical queries")
triage_agent = Agent(
    name="TriageAgent",
    instructions="Initial triage of user",
    handoffs=[handoff(medical_agent)],
    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
)

result = triage_agent.run("I have a headache and fever for two days")
print(result.content)
