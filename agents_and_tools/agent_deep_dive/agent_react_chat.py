"""
Groq-Compatible LangChain Agent
- Explicit Tool Schemas
- Time tool
- Wikipedia search tool
- Conversation memory
- Chat loop
"""

from dotenv import load_dotenv
load_dotenv()  # GROQ_API_KEY

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
import datetime
from wikipedia import summary

# -----------------------------
# Tools
# -----------------------------
def get_current_time_func() -> str:
    """Returns current time H:MM AM/PM"""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

get_current_time_tool = Tool(
    name="get_current_time",
    func=get_current_time_func,
    description="Returns the current time in H:MM AM/PM format",
    args_schema={}  # No arguments
)

def search_wikipedia_func(query: str) -> str:
    """Search Wikipedia and return a short summary"""
    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."

search_wikipedia_tool = Tool(
    name="search_wikipedia",
    func=search_wikipedia_func,
    description="Searches Wikipedia for a topic and returns a short summary",
    args_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
)

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -----------------------------
# Agent
# -----------------------------
agent = create_agent(
    model=llm,
    tools=[get_current_time_tool, search_wikipedia_tool],
    system_prompt=(
        "You are a helpful AI assistant. Use tools when needed. "
        "After calling a tool, respond with the tool's result clearly."
    )
)

# -----------------------------
# Chat Memory
# -----------------------------
messages = [
    ("system", "You are an AI assistant that can answer questions using tools like Time and Wikipedia.")
]

# -----------------------------
# Chat Loop
# -----------------------------
def main():
    print("Type 'exit' or Ctrl+C to quit.\n")
    try:
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            messages.append(("user", user_input))

            response = agent.invoke({"messages": messages})

            final_message = response["messages"][-1].content
            print("\n🟢 Bot Answer:\n", final_message, "\n")

            messages.append(("assistant", final_message))

    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
