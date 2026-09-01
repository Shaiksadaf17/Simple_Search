from vespa.application import Vespa
from sentence_transformers import CrossEncoder


# ============================================================
# STAGE 4C - RERANKING
# ============================================================

print("===================================")
print("STAGE 4C - RERANKING")
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
# 2. Load reranking model
# ------------------------------------------------------------

print("\nLoading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded.")


# ------------------------------------------------------------
# 3. Load embedding model
# ------------------------------------------------------------

print("\nLoading embedding model...")

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ------------------------------------------------------------
# 4. Hybrid retrieval
# ------------------------------------------------------------

def retrieve_documents(query, hits=20):

    query_embedding = embedding_model.encode(
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

def rerank_documents(query, documents, top_k=5):

    pairs = []

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


    # Ask the reranker to score
    # each question-document pair.

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
                "rerank_score": float(score),
            }
        )


    # Highest score first

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )


    return reranked[:top_k]


# ------------------------------------------------------------
# 6. Test query
# ------------------------------------------------------------

query = "What is machine learning?"

print("\n===================================")
print("QUERY")
print("===================================")

print(query)


# ------------------------------------------------------------
# 7. Retrieve 20 candidates
# ------------------------------------------------------------

print("\nRetrieving documents from Vespa...")

documents = retrieve_documents(
    query,
    hits=20,
)

print(
    f"Retrieved {len(documents)} candidates."
)


# ------------------------------------------------------------
# 8. Rerank candidates
# ------------------------------------------------------------

print("\nReranking documents...")

results = rerank_documents(
    query,
    documents,
    top_k=5,
)


# ------------------------------------------------------------
# 9. Display reranked results
# ------------------------------------------------------------

print("\n===================================")
print("TOP RERANKED RESULTS")
print("===================================\n")


for index, result in enumerate(
    results,
    start=1,
):

    document = result["document"]

    score = result["rerank_score"]

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
        "",
    )

    print(
        f"{index}. "
        f"Rerank Score: {score:.4f}"
    )

    print(
        f"URL: {url}"
    )

    print(
        f"Text: {text[:500]}..."
    )

    print("-" * 80)


print("\n===================================")
print("STAGE 4C COMPLETE")
print("===================================")