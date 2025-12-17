import os
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splits raw text based on character count
from langchain_community.document_loaders import TextLoader # Reads .txt files - Wraps content into LangChain Document objects.
                                                            # - Each Documentcontains: page_content → actual text and metadata → source information (e.g., file path) 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # embedding model wrapper. Converts text chunks into dense numerical vectors.

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__)) # __file__ → current script location - os.path.abspath() → converts to full path - os.path.dirname() → extracts directory only
file_path = os.path.join(current_dir, "books", "odyssey.txt") # Constructs the full path to your text file. os.path.join() ensures cross-platform compatibility.
persistent_directory = os.path.join(current_dir, "db", "chroma_db") # Defines where Chroma will persist its vector data.

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path, encoding="utf-8") # load the file but Does not read the file yet.
    documents = loader.load() # Reads the file from disk.Returns a list of Document objects.

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )# Creates a chunking strategy:
    docs = text_splitter.split_documents(documents) # Splits each document into multiple smaller Document chunks.

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")

    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i

    # Initializes the embedding model.
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )  # Update to a valid embedding model if needed

    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
    db.persist()
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")