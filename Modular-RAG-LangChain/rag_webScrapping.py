import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Embedding model (MUST be the same for indexing and querying)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Target website
TARGET_URL = "https://apple.com"

# Directory setup for persistent vector storage
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "db")
PERSISTENT_DIRECTORY = os.path.join(DB_DIR, "chroma_db_firecrawl")

# Ensure database directory exists
os.makedirs(DB_DIR, exist_ok=True)


# --------------------------------------------------
# Vector Store Creation
# --------------------------------------------------

def create_vector_store():
    """
    Crawls the target website, splits content into chunks,
    generates embeddings, and persists them in a Chroma vector store.
    """

    # Retrieve Firecrawl API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    print("Starting website crawl...")

    # Initialize Firecrawl loader
    loader = FireCrawlLoader(
        api_key=api_key,
        url=TARGET_URL,
        mode="scrape"  # Use 'scrape' for single-page, 'crawl' for deeper coverage
    )

    # Load documents from the website
    documents = loader.load()

    print(f"Crawling completed. Retrieved {len(documents)} documents.")

    # Normalize metadata to ensure compatibility with Chroma
    for doc in documents:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200  # Improves semantic continuity across chunks
    )

    split_documents = text_splitter.split_documents(documents)

    print(f"Total document chunks created: {len(split_documents)}")
    print("Sample chunk preview:\n")
    print(split_documents[0].page_content[:500])
    print("\n-----------------------------------")

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Create and persist the Chroma vector store
    print("Creating and persisting vector store...")

    Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=PERSISTENT_DIRECTORY
    )

    print(f"Vector store successfully created at:\n{PERSISTENT_DIRECTORY}")


# --------------------------------------------------
# Initialize Vector Store (Only Once)
# --------------------------------------------------

if not os.path.exists(PERSISTENT_DIRECTORY):
    create_vector_store()
else:
    print("Existing vector store found. Skipping initialization.")


# --------------------------------------------------
# Load Vector Store for Querying
# --------------------------------------------------

# Load embeddings (must match indexing embeddings)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

# Load persisted Chroma database
db = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embeddings
)


# --------------------------------------------------
# Query Function
# --------------------------------------------------

def query_vector_store(query: str):
    """
    Queries the vector store using semantic search and
    returns the most relevant document chunks.
    """

    # Configure retriever with MMR for better result diversity
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,        # Number of final results
            "fetch_k": 15  # Candidate pool for MMR
        }
    )

    # Perform retrieval
    results = retriever.invoke(query)

    print("\n--- Retrieved Relevant Documents ---\n")

    for idx, doc in enumerate(results, start=1):
        print(f"Result {idx}:")
        print(doc.page_content)
        print("\nMetadata:", doc.metadata)
        print("-" * 80)


# --------------------------------------------------
# Example Query
# --------------------------------------------------

user_query = "What is Apple Intelligence?"
query_vector_store(user_query)
