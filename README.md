# 🔎 Simple Search Engine

> A lightweight search engine built from scratch using **Python, Vespa, Docker, and FineWeb**, with **BM25** powering lexical retrieval and ranking.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Vespa-Search%20Engine-orange" />
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?logo=docker" />
  <img src="https://img.shields.io/badge/BM25-Lexical%20Search-green" />
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" />
</p>

---

## 🚀 About

**Simple Search Engine** is a practical information-retrieval project designed to understand how modern search systems work under the hood.

The current version uses **FineWeb documents**, indexes them with **Vespa**, and retrieves relevant documents using **BM25**.

The project will gradually evolve from traditional keyword search into a more modern AI-powered retrieval system.

---

## ⚡ How It Works

```text
             🌐 FineWeb
                 │
                 ▼
            🐍 Python
                 │
                 ▼
             📦 pyVespa
                 │
                 ▼
          🔎 Vespa Search
                 │
                 ▼
             📚 Index
                 │
                 ▼
            ⚡ BM25
                 │
                 ▼
         📊 Ranked Results
🛠️ Tech Stack
Technology	Role
🐍 Python	Main development language
🔎 Vespa	Search, indexing & ranking
📦 pyVespa	Python interface for Vespa
🐳 Docker	Runs Vespa locally
🐧 WSL 2	Linux environment for Docker
🤗 Hugging Face	Dataset access
🌐 FineWeb	Web document dataset
⚡ BM25	Lexical retrieval & ranking
🔧 cURL	Search API testing
🌿 Git & GitHub	Version control
📂 Project Structure
Simple_Search/
│
├── 📄 main.py
├── ⚙️ pyproject.toml
├── 🔎 query.json
├── 🚫 .gitignore
└── 📖 README.md
main.py

The main application responsible for:

Creating the Vespa application
Defining the document schema
Configuring BM25
Loading FineWeb
Preparing documents
Feeding documents into Vespa
query.json

Contains an example search request for testing the Vespa API.

🔎 Search Example

Example query:

machine learning

The query is sent to Vespa and processed using the BM25 ranking profile.

User Query
    │
    ▼
"machine learning"
    │
    ▼
   BM25
    │
    ▼
Relevant Documents
    │
    ▼
Ranked Search Results
🧠 BM25

The current search engine uses BM25, a classic information-retrieval algorithm.

Instead of simply checking whether a word exists in a document, BM25 calculates how relevant a document is to the user's query.

For example:

Query: "machine learning"

Document A → ⭐⭐⭐⭐⭐
Document B → ⭐⭐⭐
Document C → ⭐

The documents are then ranked according to their relevance scores.

The current ranking function combines the relevance of the document text with a smaller contribution from the URL:

Final Score =
    BM25(text)
    +
    0.1 × BM25(url)
🐳 Docker Architecture

Vespa runs locally inside Docker.

💻 Windows
    │
    ▼
🐧 WSL 2
    │
    ▼
🐳 Docker Desktop
    │
    ▼
🔎 Vespa Container
    │
    ▼
📚 Search Index

This makes it possible to run the search infrastructure locally without installing Vespa directly into Windows.

📊 Current Status
✅ Stage 1 — BM25 Lexical Search
 Python project setup
 Virtual environment
 Git & GitHub
 Docker & WSL 2
 Vespa deployment
 Document schema
 FineWeb integration
 Document ingestion
 BM25 indexing
 Ranking profile
 Search API testing

Current stage: 🟢 Working

🛣️ Roadmap

The project will gradually move from traditional search toward modern AI-powered retrieval.

BM25 Search
     │
     ▼
Semantic Search
     │
     ▼
Hybrid Retrieval
     │
     ▼
Reranking
     │
     ▼
Evaluation
     │
     ▼
RAG
     │
     ▼
🤖 AI-Powered Search
🔄 Stage 2 — Semantic Search
 Generate document embeddings
 Generate query embeddings
 Add vector search
 Compare semantic search with BM25
🔄 Stage 3 — Hybrid Retrieval
 Combine BM25 + vector search
 Experiment with ranking weights
 Improve retrieval quality
🔄 Stage 4 — Reranking
 Retrieve top-K candidates
 Add a reranking model
 Compare ranking performance
🔄 Stage 5 — Evaluation
 Build evaluation dataset
 Measure Recall@K
 Measure MRR
 Measure nDCG
 Compare retrieval approaches
🔄 Stage 6 — RAG
 Retrieve relevant documents
 Pass retrieved context to an LLM
 Generate grounded answers
 Evaluate answer quality
🎯 Final Vision

The final goal is to build an end-to-end retrieval system:

                    👤 User
                      │
                      ▼
                 🔎 Query
                      │
             ┌────────┴────────┐
             ▼                 ▼
          BM25             Embeddings
             │                 │
             └────────┬────────┘
                      ▼
              🔀 Hybrid Search
                      │
                      ▼
                  🏆 Reranker
                      │
                      ▼
               📊 Evaluation
                      │
                      ▼
                📚 Context
                      │
                      ▼
                    🤖 LLM
                      │
                      ▼
              💬 Final Answer
💡 What This Project Demonstrates

This project provides hands-on experience with:

Information Retrieval
Search Engineering
BM25
Document Indexing
Ranking
Semantic Search
Vector Retrieval
Hybrid Search
Reranking
Search Evaluation
Retrieval-Augmented Generation
Docker
Python
Vespa
👨‍💻 Author
Shaik Sadaf Patel

🎓 MSc Artificial Intelligence

Building and learning the foundations of modern AI-powered search systems.

⭐ If you find this project interesting, feel free to explore the code and follow its development.


This will look **much better on your GitHub profile** because it has a clear title, badges,
