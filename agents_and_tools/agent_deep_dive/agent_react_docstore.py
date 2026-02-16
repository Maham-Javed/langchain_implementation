"""
Groq-Compatible RAG + ReAct Agent (LangChain 1.0+)
- Uses Chroma vector store for retrieval
- ReAct agent with tools
- Conversation memory
- Chat loop
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Ensure GROQ_API_KEY is set

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma


# -----------------------------
# 1️⃣ Load Chroma Vector Store
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Create the directory if it doesn't exist
if not os.path.exists(persistent_directory):
    print(f"Directory {persistent_directory} not found. Creating it...")
    os.makedirs(persistent_directory, exist_ok=True)

# Initialize Chroma vector store
db = Chroma(persist_directory=persistent_directory, embedding_function=None)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------------
# 2️⃣ Initialize Groq LLM
# -----------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# -----------------------------
# 3️⃣ Define Tools
# -----------------------------
def answer_from_docs(input: str, chat_history: list[dict] | None = None) -> str:
    """
    Retrieve top-k documents from Chroma and return their combined content as an answer.
    """
    docs = retriever.get_relevant_documents(input)
    if not docs:
        return "I could not find relevant information in the documents."
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    return combined_text

answer_docs_tool = Tool(
    name="answer_from_docs",
    func=answer_from_docs,
    description="Answers questions based on the retrieved documents",
    args_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string"},
            "chat_history": {"type": "array"}
        },
        "required": ["input"]
    }
)

# -----------------------------
# 4️⃣ Create ReAct Agent
# -----------------------------
system_prompt = (
    "You are a helpful AI assistant. Use tools when needed. "
    "If you call a tool, respond with the tool output clearly."
)

agent = create_agent(
    model=llm,
    tools=[answer_docs_tool],
    system_prompt=system_prompt
)

# -----------------------------
# 5️⃣ Conversation Memory
# -----------------------------
messages = [
    ("system", "You are an AI assistant that can answer questions using documents and tools like Answer Question.")
]

# -----------------------------
# 6️⃣ Chat Loop
# -----------------------------
def main():
    print("Type 'exit' to quit.\n")
    try:
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

    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
