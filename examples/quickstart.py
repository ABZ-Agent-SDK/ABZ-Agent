import os
# Assuming you have installed the necessary dependencies (pydantic, abzagent, etc.)
from pydantic import Field
from typing import Optional

# This import should now work correctly because you fixed the Agent class in site-packages
from abzagent import Agent, function_tool

# --- 1. Define the Tool Function ---

# We use @function_tool to automatically create the necessary schema and metadata 
# from the Python type hints and docstrings.
@function_tool(
    name_override="add_numbers",
    description_override="Adds two numbers together. Use this tool for simple addition.",
)
def add(a: float = Field(description="The first number to add."),
        b: float = Field(description="The second number to add.")) -> str:
    """
    Performs addition of two floating-point numbers. Returns the result as a string.
    """
    try:
        result = a + b
        return f"The sum of {a} and {b} is {result}."
    except TypeError as e:
        return f"[Tool Execution Error] Invalid input for add_numbers: {e}"

# --- 2. Create and Configure the Agent ---

# The Agent will automatically wrap the raw function 'add' into a proper Tool object
# using the corrected logic in agent.py.
try:
    math_agent = Agent(
        name="Math Agent",
        instructions="You are an expert at math. You MUST use the 'add_numbers' tool for any addition task.",
        model="gemini-2.5-flash", # A recommended model for reliable tool calling
        tools=[add],              # This is the list that was causing the error
        verbose=True              # Prints the internal thought steps
    )
    
    print("-" * 50)
    print(f"Agent '{math_agent.name}' initialized successfully with tool: {list(math_agent.tools.keys())}")
    print("-" * 50)

    # --- 3. Run a query that requires the tool ---
    user_query = "What is the result of 45 + 12.5?"
    print(f"\n[USER]: {user_query}")
    
    # The agent will internally decide to call the 'add_numbers' tool
    result = math_agent.run(user_query)
    
    print("\n" + "=" * 50)
    print(f"[FINAL RESULT]: {result.content}")
    print(f"[STEPS TAKEN]: {len(result.steps)}")
    print("=" * 50)

    # --- 4. Run a query that does NOT require the tool ---
    user_query_2 = "Hello, what is your name and what can you do?"
    print(f"\n[USER]: {user_query_2}")
    
    # The agent should respond directly without a tool call
    result_2 = math_agent.run(user_query_2)
    
    print("\n" + "=" * 50)
    print(f"[FINAL RESULT]: {result_2.content}")
    print(f"[STEPS TAKEN]: {len(result_2.steps)}")
    print("=" * 50)


except RuntimeError as e:
    print(f"\nFATAL ERROR: {e}")
    print("Please ensure your GEMINI_API_KEY is set in your environment.")
except Exception as e:
    print(f"\nAn error occurred during execution: {e}")