from vespa.application import Vespa
from sentence_transformers import SentenceTransformer


# ============================================================
# STAGE 4A - RETRIEVAL EVALUATION
# ============================================================

print("Starting Stage 4A - Retrieval Evaluation...")


# ------------------------------------------------------------
# 1. Connect to existing Stage 3 Vespa
# ------------------------------------------------------------

print("Connecting to Vespa...")

app = Vespa(
    url="http://localhost:8080"
)

print("Connected to Vespa.")


# ------------------------------------------------------------
# 2. Load embedding model
# ------------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ------------------------------------------------------------
# 3. Hybrid search
# ------------------------------------------------------------

def hybrid_search(query, hits=5):

    # Convert query into an embedding
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
# 4. Test questions
# ------------------------------------------------------------

test_questions = [

    "How do computers learn from data?",

    "What is computer security?",

    "How does climate research use data?",

    "How can energy consumption be tracked?",

    "What is machine learning?",
]


# ------------------------------------------------------------
# 5. Evaluate each question
# ------------------------------------------------------------

for question_number, query in enumerate(
    test_questions,
    start=1,
):

    print("\n")
    print("=" * 80)
    print(f"TEST QUESTION {question_number}")
    print("=" * 80)

    print(f"\nQuery: {query}")

    response = hybrid_search(
        query,
        hits=5,
    )

    if not response.is_successful():

        print("\nSearch failed.")

        print(response.get_json())

        continue

    root = response.get_json()["root"]

    results = root.get(
        "children",
        [],
    )

    print(
        f"\nRetrieved documents: {len(results)}"
    )

    print("\nTop results:\n")


    # --------------------------------------------------------
    # Display retrieved documents
    # --------------------------------------------------------

    for index, hit in enumerate(
        results,
        start=1,
    ):

        fields = hit.get(
            "fields",
            {},
        )

        score = hit.get(
            "relevance",
            0,
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
            f"{index}. Score: {score:.4f}"
        )

        print(
            f"URL: {url}"
        )

        print(
            f"Text: {text[:400]}..."
        )

        print("-" * 80)


print("\n")
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

print(
    "\nReview the results above and check whether "
    "the retrieved documents are relevant to each question."
)