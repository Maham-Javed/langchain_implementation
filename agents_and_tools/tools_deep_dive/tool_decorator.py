"""
Groq Tool-Calling Agent Example
LangChain 1.0+ with ChatGroq
Supports multiple tools: greet, reverse, concatenate
"""

# ----------------------------------
# 1️⃣ Load Environment Variables
# ----------------------------------
from dotenv import load_dotenv
load_dotenv()  # Ensure GROQ_API_KEY is set

# ----------------------------------
# 2️⃣ Imports
# ----------------------------------
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# ----------------------------------
# 3️⃣ Define Tools
# ----------------------------------
# Simple tool without args_schema
@tool
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

# Tool with one parameter (using decorator input)
@tool
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

# Tool with two parameters
@tool
def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b

# ----------------------------------
# 4️⃣ Initialize Groq LLM
# ----------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ----------------------------------
# 5️⃣ Create Agent
# ----------------------------------
system_prompt = (
    "You are an AI assistant that MUST use tools when available. "
    "After calling a tool, respond clearly using the tool result."
)

agent = create_agent(
    model=llm,
    tools=[greet_user, reverse_string, concatenate_strings],
    system_prompt=system_prompt
)

# ----------------------------------
# 6️⃣ Interactive Chat Loop
# ----------------------------------
if __name__ == "__main__":
    print("Type 'exit' to quit.\n")
    messages = [("system", system_prompt)]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        messages.append(("user", user_input))

        response = agent.invoke({"messages": messages})
        final_message = response["messages"][-1].content

        print("\n🟢 AI Answer:\n", final_message, "\n")
        messages.append(("assistant", final_message))
