import requests

from vespa.application import Vespa
from sentence_transformers import SentenceTransformer


# ============================================================
# STAGE 3 - RAG
#
# 3A - Hybrid Search
# 3B - Ollama + Gemma 3
# 3C - RAG Prompt
# 3D - Final Answer
# ============================================================


# ------------------------------------------------------------
# 1. Connect to the existing Vespa application
# ------------------------------------------------------------

print("Connecting to Vespa...")

app = Vespa(url="http://localhost:8080")

print("Connected to Vespa.")


# ------------------------------------------------------------
# 2. Load the embedding model
# ------------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ------------------------------------------------------------
# 3. Hybrid search
# ------------------------------------------------------------

def hybrid_search(query, hits=5):

    # Convert the user's question into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    response = app.query(
        body={
            "yql": (
                "select * from sources * "
                "where ({targetHits:100}"
                "nearestNeighbor(embedding, query_embedding)) "
                "or userQuery();"
            ),

            "query": query,

            "input.query(query_embedding)": query_embedding,

            "ranking": "hybrid",

            "hits": hits,
        }
    )

    return response


# ------------------------------------------------------------
# 4. Extract documents from Vespa
# ------------------------------------------------------------

def get_documents(response):

    documents = []

    if not response.is_successful():
        print("Vespa search failed.")
        print(response.get_json())
        return documents

    root = response.get_json()["root"]

    for hit in root.get("children", []):

        fields = hit.get("fields", {})

        text = fields.get("text", "")
        url = fields.get("url", "")

        if text:

            documents.append(
                {
                    "text": text,
                    "url": url,
                }
            )

    return documents


# ------------------------------------------------------------
# 5. Create the RAG prompt
# ------------------------------------------------------------

def create_rag_prompt(query, documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        context_parts.append(
            f"""
Document {index}:

{document["text"]}

Source:
{document["url"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the provided context.

Rules:
- Use the context as your main source of information.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, say that you do not have enough information.
- Give a clear and concise answer.

Context:
{context}

User question:
{query}

Answer:
"""

    return prompt


# ------------------------------------------------------------
# 6. Send the RAG prompt to Ollama
# ------------------------------------------------------------

def ask_ollama(prompt):

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "gemma3",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


# ------------------------------------------------------------
# 7. Run the complete RAG pipeline
# ------------------------------------------------------------

def rag_search(query):

    print("\n===================================")
    print("1. HYBRID SEARCH")
    print("===================================")

    print(f"Query: {query}")

    # Retrieve documents from Vespa
    response = hybrid_search(
        query,
        hits=5,
    )

    documents = get_documents(response)

    print(
        f"Retrieved documents: {len(documents)}"
    )


    # --------------------------------------------------------
    # Create RAG prompt
    # --------------------------------------------------------

    print("\n===================================")
    print("2. RAG PROMPT")
    print("===================================")

    prompt = create_rag_prompt(
        query,
        documents,
    )

    print("RAG prompt created.")


    # --------------------------------------------------------
    # Send prompt to Gemma through Ollama
    # --------------------------------------------------------

    print("\n===================================")
    print("3. OLLAMA + GEMMA 3")
    print("===================================")

    answer = ask_ollama(prompt)


    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    print("\n===================================")
    print("4. FINAL ANSWER")
    print("===================================")

    print(answer)

    return answer


# ------------------------------------------------------------
# 8. Test the RAG system
# ------------------------------------------------------------

if __name__ == "__main__":

    query = "How do computers learn from data?"

    rag_search(query)