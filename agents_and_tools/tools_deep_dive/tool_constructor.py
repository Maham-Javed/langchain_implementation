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
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

# ----------------------------------
# 3️⃣ Define Tool Functions
# ----------------------------------
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b

# ----------------------------------
# 4️⃣ Define Tools (Explicit Tool Objects)
# ----------------------------------
greet_tool = Tool(
    name="GreetUser",
    func=greet_user,
    description="Greets the user by name. Input: name (string).",
    args_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
)

reverse_tool = Tool(
    name="ReverseString",
    func=reverse_string,
    description="Reverses the given string. Input: text (string).",
    args_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
)

concatenate_tool = Tool(
    name="ConcatenateStrings",
    func=concatenate_strings,
    description="Concatenates two strings. Input: a (string), b (string).",
    args_schema={
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"}
        },
        "required": ["a", "b"]
    }
)

tools = [greet_tool, reverse_tool, concatenate_tool]

# ----------------------------------
# 5️⃣ Initialize Groq LLM
# ----------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ----------------------------------
# 6️⃣ Create Agent
# ----------------------------------
system_prompt = (
    "You are an AI assistant that MUST use tools when available. "
    "After calling a tool, respond clearly using the tool result."
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# ----------------------------------
# 8️⃣ Interactive Chat Loop
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
