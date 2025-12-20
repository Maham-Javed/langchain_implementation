import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------

# Base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory containing text files (books)
BOOKS_DIR = os.path.join(BASE_DIR, "books")

# Directory for vector database persistence
DB_DIR = os.path.join(BASE_DIR, "db")
PERSIST_DIRECTORY = os.path.join(DB_DIR, "chroma_db_with_metadata")

print(f"Books directory: {BOOKS_DIR}")
print(f"Persistent directory: {PERSIST_DIRECTORY}")


# ---------------------------------------------------------
# Embedding Model Configuration
# ---------------------------------------------------------

# IMPORTANT:
# This embedding model MUST be the same for ingestion and retrieval
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ---------------------------------------------------------
# INGESTION: Create Vector Store (Run Once)
# ---------------------------------------------------------

if not os.path.exists(PERSIST_DIRECTORY):
    print("\nPersistent directory does not exist. Initializing vector store...")

    # Validate books directory
    if not os.path.exists(BOOKS_DIR):
        raise FileNotFoundError(
            f"The directory {BOOKS_DIR} does not exist."
        )

    # Collect all .txt files
    book_files = [
        f for f in os.listdir(BOOKS_DIR)
        if f.endswith(".txt")
    ]

    if not book_files:
        raise ValueError("No .txt files found in books directory.")

    documents = []

    # Load each book and attach metadata
    for book_file in book_files:
        file_path = os.path.join(BOOKS_DIR, book_file)

        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()

        for doc in book_docs:
            # Attach source metadata for traceability
            doc.metadata = {
                "source": book_file
            }
            documents.append(doc)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunking Info ---")
    print(f"Total chunks created: {len(docs)}")

    # Create and persist Chroma vector store
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    db.persist()

    print("--- Vector store created successfully ---")

else:
    print("\nVector store already exists. Skipping ingestion.")


# ---------------------------------------------------------
# RETRIEVAL: Load Vector Store and Query
# ---------------------------------------------------------

# Load existing Chroma DB
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# User query
query = "How did Juliet die?"

# Configure retriever
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.5
    }
)

# Execute retrieval
relevant_docs = retriever.invoke(query)


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

print("\n--- Relevant Documents ---")

if not relevant_docs:
    print("No relevant documents found. Try lowering the score threshold.")
else:
    for i, doc in enumerate(relevant_docs, start=1):
        print(f"Document {i}:\n{doc.page_content}\n")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
