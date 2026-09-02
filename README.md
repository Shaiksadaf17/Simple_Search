# 🔎 Simple Search AI

> An end-to-end AI-powered search and question-answering system built with **Python, Vespa, Docker, FineWeb, BM25, semantic search, hybrid retrieval, reranking, RAG, Ollama, Gemma 3, and Streamlit**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Vespa-Search%20Engine-orange" />
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?logo=docker" />
  <img src="https://img.shields.io/badge/BM25-Lexical%20Search-green" />
  <img src="https://img.shields.io/badge/Semantic%20Search-Embeddings-purple" />
  <img src="https://img.shields.io/badge/Reranking-Cross--Encoder-red" />
  <img src="https://img.shields.io/badge/RAG-Gemma%203-yellow" />
  <img src="https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit" />
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" />
</p>

---

## 🚀 About

**Simple Search AI** started as a lightweight BM25 search engine and has been progressively developed into a multi-stage **retrieval and RAG system**.

The project is designed to understand how modern AI search systems work from the ground up:

```text
Traditional Search
      ↓
BM25
      ↓
Semantic Search
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
Retrieval Evaluation
      ↓
RAG + Citations
      ↓
Streamlit AI Search UI
```

The system currently uses documents from the **FineWeb** dataset, indexes them in **Vespa**, retrieves candidates using lexical and semantic search, reranks them with a cross-encoder, and uses **Ollama + Gemma 3** to generate a grounded answer from the retrieved sources.

---

## 🖥️ Main Project UI

The main user-facing application is built with **Streamlit**.

The current UI allows the user to enter a question and run the complete pipeline:

```text
┌──────────────────────────────────────────────────────┐
│ 🔎 Simple Search AI                                  │
│                                                      │
│ Hybrid Search + Reranking + RAG + Gemma 3           │
│                                                      │
│ Ask a question                                      │
│ ┌──────────────────────────────────────────────────┐ │
│ │ What is machine learning?                       │ │
│ └──────────────────────────────────────────────────┘ │
│                                                      │
│                  🔍 Search                           │
│                                                      │
│ 💬 Answer                                            │
│                                                      │
│ The answer generated from the retrieved sources... │
│                                                      │
│ 📚 Sources                                           │
│ ├── Source 1 — Rerank score                         │
│ ├── Source 2 — Rerank score                         │
│ └── Source 3 — Rerank score                         │
└──────────────────────────────────────────────────────┘
```

### UI pipeline

```text
User Question
     │
     ▼
Hybrid Search
     │
     ▼
20 Candidate Documents
     │
     ▼
Cross-Encoder Reranking
     │
     ▼
Top 5 Documents
     │
     ▼
RAG Context
     │
     ▼
Ollama + Gemma 3
     │
     ▼
💬 Grounded Answer
     │
     ▼
📚 Source URLs + Rerank Scores
```

---

## ⚡ How It Works

### 1. Document ingestion

FineWeb documents are loaded through Hugging Face Datasets and prepared for Vespa.

Each document contains:

- `id`
- `text`
- `url`

The documents are then fed into a Vespa application.

### 2. BM25 lexical search

The first version of the project uses **BM25** to retrieve documents based on query terms.

The Vespa ranking profile combines the text and URL relevance:

```text
Final Score =
    BM25(text)
    +
    0.1 × BM25(url)
```

### 3. Semantic search

The project then adds vector search using:

```text
all-MiniLM-L6-v2
```

The model converts documents and queries into **384-dimensional embeddings**.

Vespa stores the embeddings and performs nearest-neighbor vector retrieval.

### 4. Hybrid retrieval

The project combines:

```text
BM25 + Semantic Vector Search
```

This allows the search engine to use both:

- Exact keyword matching
- Semantic similarity

### 5. Reranking

Hybrid search initially retrieves a larger candidate set.

The candidates are then passed through:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder scores the query-document pairs and the best five documents are selected.

```text
Hybrid Search
     ↓
Top 20 Candidates
     ↓
Cross-Encoder
     ↓
Top 5 Results
```

### 6. RAG

The top-ranked documents are converted into a context containing their text and URLs.

That context is passed to a RAG prompt and sent to:

```text
Ollama
   ↓
Gemma 3
```

The model is instructed to answer using only the retrieved sources.

### 7. Citations

The RAG pipeline includes source labels such as:

```text
[Source 1]
[Source 2]
[Source 3]
```

The Streamlit UI also displays the source URL and rerank score for each retrieved document.

---

## 🧠 System Architecture

```text
                         👤 USER
                           │
                           ▼
                     🔎 QUESTION
                           │
                           ▼
              ┌────────────────────────┐
              │     HYBRID SEARCH      │
              │                        │
              │  BM25 + Vector Search  │
              └───────────┬────────────┘
                          │
                          ▼
                  Candidate Documents
                          │
                          ▼
                🏆 CROSS-ENCODER
                   RERANKING
                          │
                          ▼
                    Top 5 Sources
                          │
                          ▼
                    📚 RAG CONTEXT
                          │
                          ▼
                  🤖 OLLAMA / GEMMA 3
                          │
                          ▼
                    💬 FINAL ANSWER
                          │
                          ▼
                    📖 CITATIONS
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| 🐍 Python | Main development language |
| 🔎 Vespa | Search, indexing and ranking |
| 📦 pyVespa | Python interface for Vespa |
| 🐳 Docker | Runs Vespa locally |
| 🐧 WSL 2 | Linux environment for Docker on Windows |
| 🤗 Hugging Face Datasets | Dataset loading |
| 🌐 FineWeb | Web document dataset |
| ⚡ BM25 | Lexical retrieval and ranking |
| 🧠 Sentence Transformers | Query/document embeddings |
| 🔀 Hybrid Search | BM25 + semantic retrieval |
| 🏆 Cross-Encoder | Document reranking |
| 🤖 Ollama | Local LLM runtime |
| 💎 Gemma 3 | Answer generation |
| 🎨 Streamlit | Main project UI |
| 🌿 Git & GitHub | Version control |

---

## 📂 Project Structure

```text
Simple_Search/
│
├── 📄 main.py
├── 🧠 stage2_semantic.py
├── 🔀 stage3_hybrid.py
├── 🤖 stage3_rag.py
│
├── 📊 stage4A_evaluation.py
├── 🔎 stage4B_retrieval.py
├── 🏆 stage4C_reranking.py
├── 📚 stage4D_citations.py
│
├── 🎨 app.py
├── 🔎 query.json
├── ⚙️ pyproject.toml
├── 🚫 .gitignore
└── 📖 README.md
```

---

## 📄 What Each File Does

### `main.py` — Stage 1

The original search-engine implementation.

It is responsible for:

- Creating the Vespa application
- Defining the document schema
- Indexing `text` and `url`
- Enabling BM25
- Loading FineWeb
- Preparing documents
- Feeding documents into Vespa
- Testing the search infrastructure

The initial ranking profile is:

```text
BM25(text) + 0.1 × BM25(url)
```

---

### `stage2_semantic.py` — Stage 2

Introduces **semantic/vector search**.

It:

- Loads `all-MiniLM-L6-v2`
- Creates 384-dimensional embeddings
- Adds an embedding tensor to the Vespa schema
- Uses Vespa nearest-neighbor search
- Takes a development sample of 1,000 FineWeb documents
- Generates embeddings for the documents
- Tests semantic search with natural-language questions

Example:

```text
Query:
How do computers learn from data?

        ↓

Query Embedding
        ↓
Vespa Vector Search
        ↓
Semantic Results
```

---

### `stage3_hybrid.py` — Stage 3A

Introduces **hybrid retrieval**.

Instead of relying only on BM25 or only on embeddings, the system retrieves using both:

```text
BM25
 +
Semantic Vector Search
 =
Hybrid Retrieval
```

The script uses a larger development collection of **5,000 documents** and tests multiple questions to inspect retrieval behaviour.

Example test questions include:

```text
How do computers learn from data?
What is computer security?
How does climate research use data?
How can energy consumption be tracked?
What is machine learning?
```

---

### `stage3_rag.py` — Stage 3B

Adds the first **Retrieval-Augmented Generation** pipeline.

The process is:

```text
User Question
      ↓
Hybrid Retrieval
      ↓
Retrieved Documents
      ↓
RAG Context
      ↓
Prompt
      ↓
Ollama
      ↓
Gemma 3
      ↓
Generated Answer
```

This connects the retrieval system to a local language model.

---

### `stage4A_evaluation.py` — Stage 4A

Introduces retrieval evaluation experiments.

The script runs predefined questions through the retrieval system and displays the retrieved documents so retrieval quality can be inspected and compared.

The goal is to move beyond simply building search and start measuring whether the system retrieves useful documents.

---

### `stage4B_retrieval.py` — Stage 4B

Expands the retrieval experiments using a larger document collection.

It runs multiple test questions and compares the retrieved results to earlier retrieval experiments.

This stage focuses on understanding how retrieval behaves as the collection becomes larger.

---

### `stage4C_reranking.py` — Stage 4C

Adds **cross-encoder reranking**.

The system first retrieves:

```text
20 candidate documents
```

and then reranks them using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The best:

```text
Top 5 documents
```

are selected for the next stage.

```text
Query
  ↓
Hybrid Search
  ↓
20 Candidates
  ↓
Cross-Encoder
  ↓
5 Best Documents
```

This creates a multi-stage retrieval pipeline instead of relying on a single ranking method.

---

### `stage4D_citations.py` — Stage 4D

Extends the RAG system with **source-aware answers**.

It:

- Retrieves documents using hybrid search
- Reranks the candidates
- Builds a context from the top five documents
- Includes URLs in the context
- Generates a grounded answer with Gemma 3
- Uses `[Source N]` citation labels

The prompt explicitly instructs the model to avoid unsupported information and state when the retrieved sources are insufficient.

---

### `app.py` — Main Streamlit Application

This is the main user interface for the project.

It brings the major components together:

```text
Streamlit
   │
   ├── Query Input
   │
   ├── Hybrid Search
   │
   ├── Cross-Encoder Reranking
   │
   ├── RAG Context
   │
   ├── Ollama + Gemma 3
   │
   └── Source Display
```

The application:

1. Connects to Vespa
2. Loads the embedding model
3. Loads the cross-encoder reranker
4. Accepts a user question
5. Retrieves 20 hybrid-search candidates
6. Reranks them
7. Keeps the top 5
8. Builds the RAG context
9. Sends the context to Gemma 3 through Ollama
10. Displays the answer
11. Displays the retrieved source URLs and rerank scores

---

### `query.json`

Contains an example search request used for testing the Vespa API.

---

### `pyproject.toml`

Contains the Python project configuration and dependencies required by the project.

---

### `.gitignore`

Specifies files and directories that should not be committed to Git.

---

## 🖥️ Running the Project

### 1. Create and activate the virtual environment

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install the project dependencies

```bash
pip install -e .
```

### 3. Start the search system

The early stages deploy Vespa locally through Docker using `pyVespa`.

For example:

```bash
python main.py
```

or run the relevant stage as you progress through the project:

```bash
python stage2_semantic.py
python stage3_hybrid.py
python stage3_rag.py
```

### 4. Run the Streamlit UI

After Vespa and Ollama are running:

```bash
streamlit run app.py
```

Then open the Streamlit application in your browser.

---

## 🐳 Docker Architecture

Vespa runs locally using Docker.

```text
💻 Windows
    │
    ▼
🐧 WSL 2
    │
    ▼
🐳 Docker Desktop
    │
    ▼
🔎 Vespa
    │
    ▼
📚 Search Index
```

The project therefore keeps the search infrastructure local during development.

---

## 🤖 RAG Architecture

The final implemented RAG flow is:

```text
                  User Question
                       │
                       ▼
              Query Embedding
                       │
                       ▼
              ┌────────────────┐
              │ Hybrid Search  │
              │ BM25 + Vector  │
              └───────┬────────┘
                      │
                      ▼
                20 Candidates
                      │
                      ▼
               Cross-Encoder
                 Reranking
                      │
                      ▼
                  Top 5 Docs
                      │
                      ▼
                 RAG Context
                      │
                      ▼
              Grounded Prompt
                      │
                      ▼
                Ollama / Gemma 3
                      │
                      ▼
                Final Answer
                      │
                      ▼
                  Citations
```

---

## 📊 What Has Been Achieved

### ✅ Stage 1 — BM25 Search

- Python project setup
- Virtual environment
- Git & GitHub
- Docker & WSL 2
- Vespa deployment
- Document schema
- FineWeb integration
- Document ingestion
- BM25 indexing
- Ranking profile
- Search API testing

**Status: 🟢 Completed**

### ✅ Stage 2 — Semantic Search

- Sentence Transformer integration
- Document embeddings
- Query embeddings
- 384-dimensional vectors
- Vespa nearest-neighbor search
- Semantic retrieval testing

**Status: 🟢 Completed**

### ✅ Stage 3 — Hybrid Retrieval

- BM25 retrieval
- Vector retrieval
- Hybrid Vespa ranking
- Multiple retrieval queries
- Larger development document collection

**Status: 🟢 Completed**

### ✅ Stage 3 — RAG

- Hybrid retrieval
- Context construction
- RAG prompt
- Ollama integration
- Gemma 3 generation

**Status: 🟢 Completed**

### ✅ Stage 4 — Retrieval Improvements

- Retrieval experiments
- Retrieval comparison
- Cross-encoder reranking
- Top-K candidate selection
- Citation-aware RAG

**Status: 🟢 Implemented**

### ✅ Stage 5 — Streamlit UI

- User question input
- Hybrid search
- Reranking
- RAG
- Gemma 3 generation
- Answer display
- Source display
- Rerank scores

**Status: 🟢 Implemented**

---

## 🛣️ Roadmap

The project is continuing toward a more complete AI document-research system.

```text
                    CURRENT
                       │
                       ▼
               Streamlit AI Search
                       │
                       ▼
                📄 Document Upload
                       │
                       ▼
              Document Processing
                       │
                       ▼
                Document Indexing
                       │
                       ▼
                Hybrid Retrieval
                       │
                       ▼
                  Reranking
                       │
                       ▼
                  RAG + Citations
                       │
                       ▼
              🤖 AI Document Researcher
```

### 🔄 Next Improvements

- Add PDF/document upload directly through the UI
- Create document-specific indexes
- Improve document chunking
- Improve citation accuracy
- Add stronger retrieval evaluation
- Measure Recall@K
- Measure MRR
- Measure nDCG
- Compare different embedding models
- Experiment with hybrid-ranking weights
- Compare different reranking models
- Compare different local LLMs
- Add conversational follow-up questions
- Add answer-quality evaluation
- Containerize the complete application

---

## 🎯 Final Vision

The long-term goal is to evolve **Simple Search AI** into an end-to-end **AI Document Researcher**.

```text
                         👤 USER
                           │
                           ▼
                    📄 UPLOAD DOCUMENT
                           │
                           ▼
                 Document Processing
                           │
                           ▼
                  🗂️ Vector + Text Index
                           │
                           ▼
                       🔎 QUERY
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
          ⚡ BM25                    🧠 Embeddings
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    🔀 HYBRID SEARCH
                           │
                           ▼
                       🏆 RERANK
                           │
                           ▼
                     📚 TOP SOURCES
                           │
                           ▼
                         🤖 LLM
                           │
                           ▼
                    💬 GROUNDED ANSWER
                           │
                           ▼
                    📖 CITATIONS
```

---

## 💡 What This Project Demonstrates

This project provides hands-on experience with:

- Information Retrieval
- Search Engineering
- BM25
- Document Indexing
- Ranking
- Vespa
- Semantic Search
- Vector Retrieval
- Hybrid Search
- Cross-Encoder Reranking
- Retrieval Evaluation
- Retrieval-Augmented Generation
- Source Citations
- Local LLMs
- Ollama
- Gemma 3
- Streamlit
- Docker
- Python
- Git & GitHub

More importantly, the project demonstrates how a search system can be built **incrementally**, moving from a traditional lexical search engine to a multi-stage AI retrieval and RAG pipeline.

---

## 👨‍💻 Author

**Shaik Sadaf Patel**

🎓 MSc Artificial Intelligence

Building and learning the foundations of modern AI-powered search and retrieval systems.

---

⭐ If you find this project interesting, feel free to explore the code and follow its development.
