import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------

# Absolute path to current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Location of the persisted Chroma vector database
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# ---------------------------------------------------------
# Safety Check
# ---------------------------------------------------------

# Ensure ingestion has been run before retrieval
if not os.path.exists(persistent_directory):
    raise FileNotFoundError(
        "Chroma DB not found. Please run the ingestion script first."
    )


# ---------------------------------------------------------
# Embedding Model (Must Match Ingestion)
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)


# ---------------------------------------------------------
# Load Existing Vector Store
# ---------------------------------------------------------

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)


# ---------------------------------------------------------
# User Query
# ---------------------------------------------------------

query = "Who is Odysseus' wife?"


# ---------------------------------------------------------
# Retriever Configuration
# ---------------------------------------------------------

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,                # Maximum number of results
        "score_threshold": 0.4 # Filter out weak matches
    },
)

# Execute similarity search
relevant_docs = retriever.invoke(query)


# ---------------------------------------------------------
# Display Retrieved Results
# ---------------------------------------------------------

print("\n--- Relevant Documents ---")

if not relevant_docs:
    print("No relevant documents found. Try lowering the score threshold.")
else:
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"Document {i}:\n{doc.page_content}\n")

        # Print metadata if available
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
