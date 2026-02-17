# Modular-RAG-LangChain

A **learning-focused, experimental Retrieval-Augmented Generation (RAG) playground** built with **LangChain**, **Chroma**, and modern open‑source LLM tooling.  
This repository is designed as a **portfolio project** and a **hands-on reference** that demonstrates multiple RAG patterns — from the most basic vector search to advanced, history‑aware conversational RAG systems.

Each script is **independent and self-contained**, allowing you to explore, run, and understand individual RAG concepts without hidden dependencies or assumptions.

---

## 🎯 Project Objectives

- Learn and experiment with **Retrieval-Augmented Generation (RAG)** concepts
- Understand how vector databases work in real AI pipelines
- Explore different retrieval strategies (similarity, thresholding, MMR)
- Build intuition around **embeddings, chunking, metadata, and memory**
- Demonstrate practical RAG implementations suitable for a **developer portfolio**

This project intentionally prioritizes **clarity, correctness, and modularity** over abstraction-heavy frameworks.

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) enhances language models by grounding their responses in **external knowledge sources**. Instead of relying purely on model memory, the system:

1. Retrieves relevant documents from a vector database
2. Injects that context into the LLM prompt
3. Generates responses based only on retrieved information

This approach reduces hallucinations and enables domain‑specific intelligence.

---

## 📁 Project Structure

```text
.
├── basic_rag_1a.py                 # Text ingestion & vector store creation
├── basic_rag_1b.py                 # Vector retrieval & similarity search
├── rag_with_metadata.py            # RAG with document source metadata
├── rag_webScrapping.py             # Web-based RAG using Firecrawl
├── rag_with_contectualMemory.py    # Conversational RAG with history awareness
├── books/                          # Text corpus (plain .txt files)
├── db/                             # Persisted Chroma vector databases
├── requirements.txt                # All Python dependencies
├── .env.example                    # Environment variable template
└── README.md
```

Each script is designed to be executed **independently**.

---

## ⚙️ Environment Setup

### Python Version

- **Python 3.10+** (required)

### Clone Repository

```bash
git clone <your-repo-url>
cd Modular-RAG-LangChain
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Some scripts require API keys. Create a `.env` file using the example below.

### `.env.example`

```env
# Required for conversational RAG
GROQ_API_KEY=your_groq_api_key_here

# Required for web scraping RAG
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

Rename `.env.example` to `.env` before running the scripts.

---

## 🧩 Script Breakdown (Standalone Modules)

---

### 1️⃣ `basic_rag_1a.py` — Text Ingestion & Vector Store Creation

**Purpose:**
- Load a local text file
- Split it into semantic chunks
- Generate embeddings
- Persist vectors using Chroma

**Key Concepts:**
- Recursive text chunking
- Embedding normalization
- Persistent vector databases

**Run:**
```bash
python basic_rag_1a.py
```

> ⚠️ This script should be run **once**. It creates the vector database.

---

### 2️⃣ `basic_rag_1b.py` — Vector Retrieval & Similarity Search

**Purpose:**
- Load an existing Chroma database
- Perform semantic similarity search

**Retrieval Strategy:**
- Similarity score threshold

**Example Query:**
```python
query = "Who is Odysseus' wife?"
```

**Run:**
```bash
python basic_rag_1b.py
```

> ⚠️ Requires `basic_rag_1a.py` to be executed first.

---

### 3️⃣ `rag_with_metadata.py` — RAG with Source Metadata

**Purpose:**
- Ingest multiple documents
- Attach metadata (e.g., source file names)
- Retrieve answers with traceable sources

**Why Metadata Matters:**
- Improves transparency
- Enables document attribution
- Essential for real-world RAG systems

**Example Query:**
```python
query = "How did Juliet die?"
```

**Run:**
```bash
python rag_with_metadata.py
```

---

### 4️⃣ `rag_webScrapping.py` — Web-Based RAG (Firecrawl)

**Purpose:**
- Scrape live web content
- Convert pages into vector embeddings
- Query website knowledge semantically

**Key Features:**
- Firecrawl-powered scraping
- Metadata normalization
- MMR-based retrieval for diversity

**Example Query:**
```python
"What is Apple Intelligence?"
```

**Run:**
```bash
python rag_webScrapping.py
```

> ⚠️ Requires `FIRECRAWL_API_KEY`.

---

### 5️⃣ `rag_with_contectualMemory.py` — Conversational RAG with Memory

**Purpose:**
- Enable multi-turn conversations
- Rewrite follow-up questions into standalone queries
- Ground responses strictly in retrieved context

**Key Concepts:**
- History-aware retrieval
- Question contextualization
- Controlled memory growth
- Hallucination reduction

**Run:**
```bash
python rag_with_contectualMemory.py
```

**Exit Chat:**
```text
exit
```

---

## 🧠 LLM & Embedding Design Decisions

### Embeddings

- **Model:** `BAAI/bge-small-en-v1.5`
- Chosen for:
  - Strong semantic performance
  - Lightweight footprint
  - Excellent retrieval accuracy

> ⚠️ The embedding model **must remain consistent** across ingestion and retrieval.

---

### Language Model

- **Provider:** Groq
- **Model:** Llama 3.1 (8B Instant)
- Used for:
  - Question rewriting
  - Final answer generation

**Temperature:** 0.7 (balanced creativity and precision)

---

## ⚠️ Important Notes & Best Practices

- Always run ingestion scripts **before retrieval scripts**
- Do not mix embedding models between indexing and querying
- Persistent databases are reused automatically
- Lower score thresholds if no results are returned
- Keep chunk sizes balanced to avoid context loss

---

## 🎓 Intended Audience

- AI / ML Engineers
- Students learning RAG systems
- Developers building LLM applications
- Recruiters reviewing practical AI portfolios

---

## 📌 Portfolio Positioning

This project demonstrates:
- Strong understanding of RAG architecture
- Practical LangChain usage
- Vector database design
- Conversational AI grounding
- Clean, modular Python engineering

---

## ✅ Final Notes

This repository is intentionally **transparent and educational**. Every design decision is explicit, and every module can be studied in isolation. It is ideal for learning, experimentation, and showcasing real-world RAG skills.

If you’re exploring RAG seriously — this is a solid foundation.

