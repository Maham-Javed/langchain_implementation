"""
Groq Tool-Calling Agent (LangChain 1.0+)

This module demonstrates how to build a production-ready tool-calling agent
using LangChain 1.0+ and Groq as the LLM provider.

Features:
- Custom tools implemented via BaseTool subclassing
- Tavily-powered web search tool
- Numerical multiplication tool
- Interactive CLI chat loop
- Environment variable configuration

Environment Variables Required:
- GROQ_API_KEY
- TAVILY_API_KEY
"""

# =============================================================================
# 1. Environment Setup
# =============================================================================

import os
from typing import Type
from dotenv import load_dotenv

# Load environment variables from .env file if present.
# Ensure GROQ_API_KEY and TAVILY_API_KEY are properly configured.
load_dotenv()


# =============================================================================
# 2. Core Imports
# =============================================================================

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


# =============================================================================
# 3. Tool Input Schemas (Pydantic Models)
# =============================================================================
# These models define structured input validation for tools.
# They ensure strong typing and improve tool-calling reliability.

class SimpleSearchInput(BaseModel):
    """
    Schema for the simple_search tool input.
    """
    query: str = Field(description="Search query for retrieving current information")


class MultiplyNumbersArgs(BaseModel):
    """
    Schema for the multiply_numbers tool input.
    """
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")


# =============================================================================
# 4. Custom Tool Implementations
# =============================================================================
# Tools are implemented by subclassing BaseTool.
# Each tool defines:
# - name
# - description
# - args_schema
# - _run() method containing business logic


class SimpleSearchTool(BaseTool):
    """
    Tool for retrieving current information using Tavily search API.
    """
    name: str = "simple_search"
    description: str = (
        "Useful for answering questions about current events or recent information."
    )
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """
        Executes a Tavily search query and returns formatted results.
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            return (
                "Error: The 'tavily-python' package is not installed. "
                "Install it using: pip install tavily-python"
            )

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY is not set in environment variables."

        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)

        return f"Search results for: {query}\n\n{results}"


class MultiplyNumbersTool(BaseTool):
    """
    Tool for performing multiplication of two numeric values.
    """
    name: str = "multiply_numbers"
    description: str = "Useful for multiplying two numeric values."
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(self, x: float, y: float) -> str:
        """
        Multiplies two numbers and returns a formatted result string.
        """
        result = x * y
        return f"The product of {x} and {y} is {result}"


# =============================================================================
# 5. Tool Registration
# =============================================================================
# Instantiate tool objects and register them with the agent.

tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]


# =============================================================================
# 6. Language Model Initialization
# =============================================================================
# Configure Groq LLM.
# temperature=0 ensures deterministic and consistent outputs.

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# =============================================================================
# 7. Agent Creation
# =============================================================================
# create_agent builds a modern tool-calling agent.
# The system prompt guides tool usage behavior.

system_prompt = (
    "You are a professional AI assistant. "
    "Use tools whenever they are relevant to the user's request. "
    "When calling a tool, return its output clearly and directly."
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# =============================================================================
# 8. Interactive CLI Application
# =============================================================================
# This section runs a continuous conversation loop.
# Conversation state is maintained via the `messages` list.

if __name__ == "__main__":

    print("Interactive Groq Agent started. Type 'exit' to terminate.\n")

    # Conversation history
    messages = [("system", system_prompt)]

    try:
        while True:
            user_input = input("User: ")

            if user_input.lower() == "exit":
                print("Session terminated.")
                break

            # Append user message to conversation history
            messages.append(("user", user_input))

            # Invoke agent with full conversation context
            response = agent.invoke({"messages": messages})

            # Extract final assistant message
            final_message = response["messages"][-1].content

            print("\nAssistant Response:\n")
            print(final_message)
            print()

            # Append assistant response to history
            messages.append(("assistant", final_message))

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
