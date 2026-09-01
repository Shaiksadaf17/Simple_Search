from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    RankProfile,
    FieldSet,
    Function,
)
from vespa.deployment import VespaDocker
from vespa.io import VespaResponse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# STAGE 2 - SEMANTIC SEARCH
# ============================================================

# ------------------------------------------------------------
# 1. Load embedding model
# ------------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDING_DIM = 384

print("Embedding model loaded.")


# ------------------------------------------------------------
# 2. Create Vespa application package
# ------------------------------------------------------------

package = ApplicationPackage(
    name="semanticsearch",
    schema=[
        Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(
                        name="id",
                        type="string",
                        indexing=["summary"],
                    ),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="url",
                        type="string",
                        indexing=["summary"],
                    ),
                    Field(
                           name="embedding",
                           type=f"tensor<float>(x[{EMBEDDING_DIM}])",
                           indexing=["attribute", "index"],
                           hnsw={
                               "distance-metric": "angular",
                           },
                    ),
                ]
            ),
            fieldsets=[
                FieldSet(
                    name="default",
                    fields=["text", "url"],
                )
            ],
            rank_profiles=[
                RankProfile(
                    name="semantic",
                    inputs=[
                        (
                            "query(query_embedding)",
                            f"tensor<float>(x[{EMBEDDING_DIM}])",
                        )
                    ],
                    first_phase="closeness(field, embedding)",
                )
            ],
        )
    ],
)


# ------------------------------------------------------------
# 3. Deploy Vespa using Docker
# ------------------------------------------------------------

print("Starting Vespa...")

vespa_docker = VespaDocker()

app = vespa_docker.deploy(
    application_package=package
)

print("Vespa deployment complete.")


# ------------------------------------------------------------
# 4. Load a small amount of FineWeb
# ------------------------------------------------------------

print("Loading FineWeb dataset...")

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
)

# IMPORTANT:
# During development we only use 1,000 documents.
dataset = dataset.take(1000)

print("Dataset ready.")


# ------------------------------------------------------------
# 5. Create embeddings and prepare Vespa documents
# ------------------------------------------------------------

def prepare_document(document):
    text = document["text"]

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    ).tolist()

    return {
        "id": document["id"],
        "fields": {
            "id": document["id"],
            "text": text,
            "url": document["url"],
            "embedding": {
                "values": embedding
            },
        },
    }


print("Creating document embeddings...")

vespa_feed = (
    prepare_document(document)
    for document in tqdm(
        dataset,
        total=1000,
        desc="Creating embeddings",
        unit="docs",
    )
)


# ------------------------------------------------------------
# 6. Feed documents into Vespa
# ------------------------------------------------------------

feed_count = {
    "success": 0,
    "error": 0,
}


def callback(response: VespaResponse, id: str):
    if response.is_successful():
        feed_count["success"] += 1
    else:
        feed_count["error"] += 1

        print(
            f"Error feeding document {id}: "
            f"{response.get_json()}"
        )


print("Feeding documents into Vespa...")

app.feed_iterable(
    vespa_feed,
    schema="doc",
    callback=callback,
)


print("\n===================================")
print("Document feeding complete")
print("===================================")
print(f"Successful: {feed_count['success']}")
print(f"Errors:     {feed_count['error']}")


# ------------------------------------------------------------
# 7. Create semantic search function
# ------------------------------------------------------------
def semantic_search(query, hits=10):

    # Convert the user's query into a 384-dimensional vector
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    response = app.query(
        body={
            "yql": (
                "select * from sources * "
                "where {targetHits:10}"
                "nearestNeighbor(embedding, query_embedding);"
            ),
            "input.query(query_embedding)": query_embedding,
            "ranking": "semantic",
            "hits": hits,
        }
    )

    return response


# ------------------------------------------------------------
# 8. Test semantic search
# ------------------------------------------------------------

query = "How do computers learn from data?"

print("\n===================================")
print("Semantic Search")
print("===================================")
print(f"Query: {query}\n")

response = semantic_search(query)

if response.is_successful():

    root = response.get_json()["root"]

    print(
        f"Total results: "
        f"{root.get('fields', {}).get('totalCount', 0)}"
    )

    print("\nTop results:\n")

    for index, hit in enumerate(
        root.get("children", []),
        start=1,
    ):

        fields = hit.get("fields", {})

        print(
            f"{index}. "
            f"Score: {hit.get('relevance')}"
        )

        print(
            f"URL: {fields.get('url', 'N/A')}"
        )

        text = fields.get("text", "")

        print(
            f"Text: {text[:300]}..."
        )

        print("-" * 80)

else:

    print("Search failed.")

    print(response.get_json())