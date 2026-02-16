"""
Simple Tool-Calling Agent Example
Using LangChain 1.0+ with Groq LLM
"""

# ----------------------------------
# 1️⃣ Load Environment Variables
# ----------------------------------
# Required if GROQ_API_KEY is stored in a .env file
from dotenv import load_dotenv
load_dotenv()

# ----------------------------------
# 2️⃣ Imports
# ----------------------------------
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import datetime

# ----------------------------------
# 3️⃣ Define a Tool
# ----------------------------------
# The @tool decorator converts this function
# into a structured callable tool for the agent.
@tool
def get_current_time() -> str:
    """
    Returns the current time in H:MM AM/PM format.
    This tool will be invoked by the agent when needed.
    """
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

# ----------------------------------
# 4️⃣ Initialize the LLM
# ----------------------------------
# temperature=0 ensures deterministic output
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ----------------------------------
# 5️⃣ Create the Agent
# ----------------------------------
# create_agent automatically builds a modern tool-calling agent
agent = create_agent(
    model=llm,
    tools=[get_current_time],
    system_prompt=(
        "You MUST use tools when available. "
        "After calling a tool, respond clearly using the tool result."
    )
)

# ----------------------------------
# 6️⃣ Run the Agent
# ----------------------------------
if __name__ == "__main__":
    response = agent.invoke({
        "messages": [("user", "What time is it?")]
    })

    # Extract the final assistant message
    final_message = response["messages"][-1]

    print("\n Final Answer:")
    print(final_message.content)
