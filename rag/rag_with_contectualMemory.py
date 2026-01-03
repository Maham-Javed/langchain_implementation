import os
from dotenv import load_dotenv

# Vector store integration
from langchain_community.vectorstores import Chroma

# Groq integrations
from langchain_groq import ChatGroq

# Message abstractions used for chat history
from langchain_core.messages import HumanMessage, AIMessage

# Prompt templates and placeholders
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Runnable primitives (core LangChain 1.x abstraction)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Hugging face embedding
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================================================
# ENVIRONMENT & PATH CONFIGURATION
# =========================================================

# Load environment variables from `.env`
# Required for OPENAI_API_KEY and other secrets
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Resolve absolute base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Location where the Chroma vector database is persisted
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "db", "chroma_db_with_metadata")


# =========================================================
# EMBEDDING MODEL CONFIGURATION
# =========================================================

# Embedding model used to convert text into vectors
# IMPORTANT:
# This model MUST match the one used during ingestion,
# otherwise similarity search results will be invalid.
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)


# =========================================================
# LOAD VECTOR STORE
# =========================================================

# Load an existing Chroma vector database from disk.
# The embedding function is required so queries can be embedded.
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)


# =========================================================
# RETRIEVER CONFIGURATION
# =========================================================

# Convert the vector store into a retriever object.
# MMR (Maximal Marginal Relevance) is used to:
# - Reduce redundancy
# - Improve diversity of retrieved documents
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,                 # Number of final documents returned
        "fetch_k": 10,          # Initial candidate pool size
        "score_threshold": 0.3  # Minimum relevance score
    }
)


# =========================================================
# LANGUAGE MODEL CONFIGURATION
# =========================================================

# Chat-based LLM used for:
# 1. Question rewriting (history awareness)
# 2. Final answer generation
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant", 
    temperature=0.7 # temperature controls the randomness or creativity of the model’s responses
)


# =========================================================
# QUESTION CONTEXTUALIZATION (HISTORY AWARENESS)
# =========================================================

# Prompt that instructs the LLM to rewrite a follow-up question
# into a standalone query using the conversation history.
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, "
            "rewrite the question into a standalone question that "
            "can be understood without the chat history. "
            "Do NOT answer the question."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

def rewrite_question(inputs: dict) -> str:
    """
    Uses the LLM to rewrite a conversational follow-up question
    into a standalone query suitable for vector retrieval.

    Inputs:
        inputs["input"]        -> current user question
        inputs["chat_history"] -> prior conversation messages

    Returns:
        A rewritten standalone question (string)
    """
    prompt = contextualize_prompt.format(
        input=inputs["input"],
        chat_history=inputs["chat_history"]
    )
    return llm.invoke(prompt).content


# History-aware retriever pipeline:
# 1. Rewrite the question
# 2. Run vector similarity search
history_aware_retriever = (
    RunnableLambda(rewrite_question)
    | retriever
)


# =========================================================
# QUESTION ANSWERING PROMPT
# =========================================================

# Prompt instructing the LLM to answer strictly using retrieved context.
# This reduces hallucination and enforces grounded answers.
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a question-answering assistant. "
            "Use ONLY the retrieved context to answer the question. "
            "If the answer is not present, say you do not know. "
            "Use a maximum of three sentences.\n\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


# =========================================================
# RAG PIPELINE (RETRIEVAL-AUGMENTED GENERATION)
# =========================================================

# Runnable-based RAG composition:
# - context: retrieved documents
# - input: original user question
# - chat_history: conversational context
#
# Flow:
#   input -> retriever -> context
#   context + input -> prompt -> LLM
rag_chain = (
    {
        "context": history_aware_retriever,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"]
    }
    | qa_prompt
    | llm
)


# =========================================================
# INTERACTIVE CHAT LOOP
# =========================================================

def continual_chat():
    """
    Starts an interactive terminal-based chat session with:
    - History-aware retrieval
    - Vector search grounding
    - Controlled context growth
    """

    print("Start chatting with the AI (type 'exit' to stop).\n")

    # Stores conversation messages as structured objects
    chat_history = []

    # Limit chat history size to avoid prompt bloat
    MAX_HISTORY = 10

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Conversation ended.")
            break

        # Execute the RAG pipeline
        result = rag_chain.invoke(
            {
                "input": user_input,
                "chat_history": chat_history
            }
        )

        answer = result.content
        print(f"\nAI: {answer}\n")

        # Update conversation history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

        # Trim old messages to maintain context window
        if len(chat_history) > MAX_HISTORY:
            chat_history = chat_history[-MAX_HISTORY:]


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    continual_chat()
