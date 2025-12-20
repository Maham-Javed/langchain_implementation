import os

# Splits text into semantically meaningful chunks using a recursive strategy
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loads plain text files and wraps them into LangChain Document objects
from langchain_community.document_loaders import TextLoader

# Chroma vector database for embedding storage and similarity search
from langchain_community.vectorstores import Chroma

# HuggingFace embedding model wrapper
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------

# Absolute path to the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the input text file
file_path = os.path.join(current_dir, "books", "odyssey.txt")

# Directory where Chroma will persist vectors
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# ---------------------------------------------------------
# Vector Store Initialization (One-Time Setup)
# ---------------------------------------------------------

# Only initialize if the vector store does not already exist
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Validate input file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load the text file into LangChain Document objects
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split documents into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Basic sanity check
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Attach a unique chunk ID to each document
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create and persist the Chroma vector store
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )

    db.persist()
    print("Vector store created and persisted successfully.")

else:
    print("Vector store already exists. Skipping initialization.")
