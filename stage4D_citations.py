import requests

from vespa.application import Vespa
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# ============================================================
# STAGE 4D - RAG WITH CITATIONS
# ============================================================


print("===================================")
print("STAGE 4D - RAG WITH CITATIONS")
print("===================================")


# ------------------------------------------------------------
# 1. Connect to Vespa
# ------------------------------------------------------------

print("\nConnecting to Vespa...")

app = Vespa(
    url="http://localhost:8080"
)

print("Connected to Vespa.")


# ------------------------------------------------------------
# 2. Load embedding model
# ------------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ------------------------------------------------------------
# 3. Load reranker
# ------------------------------------------------------------

print("\nLoading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded.")


# ------------------------------------------------------------
# 4. Retrieve documents from Vespa
# ------------------------------------------------------------

def retrieve_documents(query, hits=20):

    # Convert the user's question into an embedding.
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    response = app.query(
        body={
            "yql": (
                "select * from sources * "
                "where ({targetHits:100}"
                "nearestNeighbor("
                "embedding, "
                "query_embedding)) "
                "or userQuery();"
            ),

            "query": query,

            "input.query("
            "query_embedding)": query_embedding,

            "ranking": "hybrid",

            "hits": hits,
        }
    )

    if not response.is_successful():

        print("Vespa search failed.")

        print(response.get_json())

        return []

    root = response.get_json()["root"]

    return root.get(
        "children",
        [],
    )


# ------------------------------------------------------------
# 5. Rerank documents
# ------------------------------------------------------------

def rerank_documents(
    query,
    documents,
    top_k=5,
):

    pairs = []

    # Create question-document pairs.
    for document in documents:

        fields = document.get(
            "fields",
            {},
        )

        text = fields.get(
            "text",
            "",
        )

        pairs.append(
            [query, text]
        )


    # Calculate relevance scores.
    scores = reranker.predict(
        pairs
    )


    reranked = []

    for document, score in zip(
        documents,
        scores,
    ):

        reranked.append(
            {
                "document": document,
                "score": float(score),
            }
        )


    # Highest score first.
    reranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )


    # Keep only the best 5.
    return reranked[:top_k]


# ------------------------------------------------------------
# 6. Create RAG context
# ------------------------------------------------------------

def create_context(
    reranked_documents,
):

    context_parts = []


    for index, result in enumerate(
        reranked_documents,
        start=1,
    ):

        document = result["document"]

        fields = document.get(
            "fields",
            {},
        )

        text = fields.get(
            "text",
            "",
        )

        url = fields.get(
            "url",
            "N/A",
        )


        context_parts.append(

            f"[Source {index}]\n"
            f"URL: {url}\n"
            f"Content:\n{text[:2000]}"

        )


    return "\n\n".join(
        context_parts
    )


# ------------------------------------------------------------
# 7. Create RAG prompt
# ------------------------------------------------------------

def create_prompt(
    query,
    context,
):

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
contained in the sources below.

Rules:

1. Do not invent facts.

2. Only make claims supported by the sources.

3. Add a citation after each important claim.

4. Use citations exactly like:
   [Source 1]
   [Source 2]

5. If the sources do not provide enough information,
   clearly say that the available sources are insufficient.

6. Do not say the sources are insufficient if they
   actually provide enough information.

7. Do not repeat the same information unnecessarily.

8. Keep the answer clear and concise.

USER QUESTION:
{query}

SOURCES:
{context}

ANSWER:
"""

    return prompt


# ------------------------------------------------------------
# 8. Send prompt to Ollama
# ------------------------------------------------------------

def ask_ollama(prompt):

    response = requests.post(

        "http://localhost:11434/api/generate",

        json={

            "model": "gemma3",

            "prompt": prompt,

            "stream": False,

        },

        timeout=120,

    )


    response.raise_for_status()

    data = response.json()

    return data["response"]


# ------------------------------------------------------------
# 9. Main RAG pipeline
# ------------------------------------------------------------

query = "What is machine learning?"


# ------------------------------------------------------------
# USER QUESTION
# ------------------------------------------------------------

print("\n===================================")
print("1. USER QUESTION")
print("===================================")

print(query)


# ------------------------------------------------------------
# HYBRID SEARCH
# ------------------------------------------------------------

print("\n===================================")
print("2. HYBRID SEARCH")
print("===================================")

documents = retrieve_documents(
    query,
    hits=20,
)

print(
    f"Retrieved {len(documents)} candidates."
)


# ------------------------------------------------------------
# RERANKING
# ------------------------------------------------------------

print("\n===================================")
print("3. RERANKING")
print("===================================")

reranked_documents = rerank_documents(
    query,
    documents,
    top_k=5,
)

print(
    f"Selected {len(reranked_documents)} "
    f"best documents."
)


# ------------------------------------------------------------
# RAG CONTEXT
# ------------------------------------------------------------

print("\n===================================")
print("4. RAG CONTEXT")
print("===================================")

context = create_context(
    reranked_documents
)

print("RAG context created.")


# ------------------------------------------------------------
# RAG PROMPT
# ------------------------------------------------------------

print("\n===================================")
print("5. RAG PROMPT")
print("===================================")

prompt = create_prompt(
    query,
    context,
)

print("RAG prompt created.")


# ------------------------------------------------------------
# GEMMA 3
# ------------------------------------------------------------

print("\n===================================")
print("6. GEMMA 3")
print("===================================")

print("Sending prompt to Ollama...")

answer = ask_ollama(
    prompt
)


# ------------------------------------------------------------
# FINAL ANSWER
# ------------------------------------------------------------

print("\n===================================")
print("7. FINAL ANSWER")
print("===================================")

print(answer)


# ------------------------------------------------------------
# SOURCES
# ------------------------------------------------------------

print("\n===================================")
print("8. SOURCES")
print("===================================")


for index, result in enumerate(
    reranked_documents,
    start=1,
):

    document = result["document"]

    fields = document.get(
        "fields",
        {},
    )

    url = fields.get(
        "url",
        "N/A",
    )

    score = result["score"]


    print(
        f"[Source {index}]"
    )

    print(
        f"Rerank score: "
        f"{score:.4f}"
    )

    print(
        f"URL: {url}"
    )

    print(
        "-" * 80
    )


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n===================================")
print("STAGE 4D COMPLETE")
print("===================================")