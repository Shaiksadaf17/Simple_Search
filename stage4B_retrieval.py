from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    RankProfile,
    FieldSet,
)
from vespa.deployment import VespaDocker
from vespa.io import VespaResponse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# STAGE 4B - BETTER RETRIEVAL
#
# Goal:
# Use a larger document collection and test whether
# retrieval quality improves.
# ============================================================


# ------------------------------------------------------------
# 1. Load embedding model
# ------------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

EMBEDDING_DIM = 384

print("Embedding model loaded.")


# ------------------------------------------------------------
# 2. Create Vespa application
# ------------------------------------------------------------

print("Creating Vespa application...")


package = ApplicationPackage(
    name="improvedsearch",

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
                        indexing=[
                            "index",
                            "summary",
                        ],
                        index="enable-bm25",
                    ),

                    Field(
                        name="url",
                        type="string",
                        indexing=["summary"],
                    ),

                    Field(
                        name="embedding",
                        type=(
                            f"tensor<float>"
                            f"(x[{EMBEDDING_DIM}])"
                        ),
                        indexing=[
                            "attribute",
                            "index",
                        ],
                        hnsw={
                            "distance-metric": "angular",
                        },
                    ),
                ]
            ),

            fieldsets=[

                FieldSet(
                    name="default",
                    fields=[
                        "text",
                        "url",
                    ],
                )
            ],

            rank_profiles=[

                RankProfile(

                    name="hybrid",

                    inputs=[

                        (
                            "query(query_embedding)",
                            (
                                f"tensor<float>"
                                f"(x[{EMBEDDING_DIM}])"
                            ),
                        )
                    ],

                    first_phase=(
                        "0.5 * bm25(text) "
                        "+ "
                        "0.5 * closeness("
                        "field, embedding)"
                    ),
                )
            ],
        )
    ],
)


# ------------------------------------------------------------
# 3. Deploy Vespa
# ------------------------------------------------------------

print("Starting Vespa...")

vespa_docker = VespaDocker()

app = vespa_docker.deploy(
    application_package=package
)

print("Vespa deployment complete.")


# ------------------------------------------------------------
# 4. Load FineWeb
# ------------------------------------------------------------

print("Loading FineWeb dataset...")

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
)


# ------------------------------------------------------------
# IMPORTANT:
# Use more documents than Stage 3.
# ------------------------------------------------------------

DOCUMENT_COUNT = 5000

dataset = dataset.take(
    DOCUMENT_COUNT
)

print(
    f"Dataset ready: "
    f"{DOCUMENT_COUNT} documents."
)


# ------------------------------------------------------------
# 5. Create document embeddings
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


print(
    "\nCreating document embeddings..."
)


vespa_feed = (

    prepare_document(document)

    for document in tqdm(

        dataset,

        total=DOCUMENT_COUNT,

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


def callback(
    response: VespaResponse,
    id: str,
):

    if response.is_successful():

        feed_count["success"] += 1

    else:

        feed_count["error"] += 1

        print(
            f"Error feeding document {id}: "
            f"{response.get_json()}"
        )


print(
    "\nFeeding documents into Vespa..."
)


app.feed_iterable(

    vespa_feed,

    schema="doc",

    callback=callback,
)


print("\n===================================")
print("Document feeding complete")
print("===================================")

print(
    f"Successful: "
    f"{feed_count['success']}"
)

print(
    f"Errors:     "
    f"{feed_count['error']}"
)


# ------------------------------------------------------------
# 7. Hybrid search
# ------------------------------------------------------------

def hybrid_search(
    query,
    hits=5,
):

    # Convert question into embedding

    query_embedding = model.encode(

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
            "query_embedding)"
            : query_embedding,

            "ranking": "hybrid",

            "hits": hits,
        }
    )

    return response


# ------------------------------------------------------------
# 8. Test questions
# ------------------------------------------------------------

test_questions = [

    "How do computers learn from data?",

    "What is computer security?",

    "How does climate research use data?",

    "How can energy consumption be tracked?",

    "What is machine learning?",
]


# ------------------------------------------------------------
# 9. Run evaluation
# ------------------------------------------------------------

for question_number, query in enumerate(

    test_questions,

    start=1,
):

    print("\n")

    print("=" * 80)

    print(
        f"TEST QUESTION "
        f"{question_number}"
    )

    print("=" * 80)

    print(
        f"\nQuery: {query}"
    )


    response = hybrid_search(

        query,

        hits=5,
    )


    if not response.is_successful():

        print("\nSearch failed.")

        print(
            response.get_json()
        )

        continue


    root = response.get_json()["root"]


    results = root.get(
        "children",
        [],
    )


    print(
        f"\nRetrieved documents: "
        f"{len(results)}"
    )


    print("\nTop results:\n")


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
            f"{index}. "
            f"Score: {score:.4f}"
        )


        print(
            f"URL: {url}"
        )


        print(
            f"Text: "
            f"{text[:400]}..."
        )


        print(
            "-" * 80
        )


print("\n")

print("=" * 80)

print(
    "STAGE 4B COMPLETE"
)

print("=" * 80)

print(
    "\nCompare these results with "
    "Stage 4A."
)