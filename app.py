import requests
import streamlit as st

from vespa.application import Vespa
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# ============================================================
# STAGE 5 - STREAMLIT RAG UI
# ============================================================


# ------------------------------------------------------------
# 1. Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Simple Search AI",
    page_icon="🔎",
    layout="wide",
)


# ------------------------------------------------------------
# 2. Title
# ------------------------------------------------------------

st.title("🔎 Simple Search AI")

st.write(
    "Hybrid Search + Reranking + RAG + Gemma 3"
)


# ------------------------------------------------------------
# 3. Load models and Vespa
# ------------------------------------------------------------

@st.cache_resource
def load_system():

    # Connect to Vespa
    app = Vespa(
        url="http://localhost:8080"
    )

    # Embedding model
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Reranker
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return (
        app,
        embedding_model,
        reranker,
    )


# ------------------------------------------------------------
# 4. Load system
# ------------------------------------------------------------

with st.spinner("Loading AI search system..."):

    app, embedding_model, reranker = load_system()


# ------------------------------------------------------------
# 5. Hybrid search
# ------------------------------------------------------------

def retrieve_documents(
    query,
    hits=20,
):

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

        return []


    root = response.get_json()["root"]


    return root.get(
        "children",
        [],
    )


# ------------------------------------------------------------
# 6. Reranking
# ------------------------------------------------------------

def rerank_documents(
    query,
    documents,
    top_k=5,
):

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


    scores = reranker.predict(
        pairs
    )


    results = []


    for document, score in zip(
        documents,
        scores,
    ):

        results.append(

            {
                "document": document,
                "score": float(score),
            }

        )


    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )


    return results[:top_k]


# ------------------------------------------------------------
# 7. Create RAG context
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
            f"Content:\n"
            f"{text[:2000]}"

        )


    return "\n\n".join(
        context_parts
    )


# ------------------------------------------------------------
# 8. Create RAG prompt
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

3. Add a citation after important claims.

4. Use citations exactly like:
   [Source 1]
   [Source 2]

5. If the sources do not contain enough information,
   say that the available sources are insufficient.

6. Keep the answer clear and concise.

USER QUESTION:
{query}

SOURCES:
{context}

ANSWER:
"""

    return prompt


# ------------------------------------------------------------
# 9. Ask Gemma 3 through Ollama
# ------------------------------------------------------------

def ask_ollama(
    prompt,
):

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


# ============================================================
# USER INTERFACE
# ============================================================


st.divider()


# ------------------------------------------------------------
# 10. Question input
# ------------------------------------------------------------

query = st.text_input(
    "Ask a question",
    placeholder="e.g. What is machine learning?",
)


# ------------------------------------------------------------
# 11. Search button
# ------------------------------------------------------------

search_clicked = st.button(
    "🔍 Search",
    type="primary",
)


# ------------------------------------------------------------
# 12. Run RAG pipeline
# ------------------------------------------------------------

if search_clicked and query:

    # --------------------------------------------------------
    # Hybrid Search
    # --------------------------------------------------------

    with st.spinner(
        "Searching documents..."
    ):

        documents = retrieve_documents(
            query,
            hits=20,
        )


    if not documents:

        st.error(
            "No documents were retrieved."
        )

        st.stop()


    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    with st.spinner(
        "Reranking results..."
    ):

        reranked_documents = rerank_documents(
            query,
            documents,
            top_k=5,
        )


    # --------------------------------------------------------
    # RAG Context
    # --------------------------------------------------------

    context = create_context(
        reranked_documents
    )


    # --------------------------------------------------------
    # RAG Prompt
    # --------------------------------------------------------

    prompt = create_prompt(
        query,
        context,
    )


    # --------------------------------------------------------
    # Gemma
    # --------------------------------------------------------

    with st.spinner(
        "Gemma 3 is generating an answer..."
    ):

        try:

            answer = ask_ollama(
                prompt
            )

        except Exception as error:

            st.error(
                f"Ollama error: {error}"
            )

            st.stop()


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.divider()

    st.subheader("💬 Answer")

    st.write(answer)


    # ========================================================
    # DISPLAY SOURCES
    # ========================================================

    st.subheader("📚 Sources")


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


        with st.expander(
            f"Source {index} — "
            f"Rerank score: {score:.4f}"
        ):

            st.write(
                f"**URL:** {url}"
            )


            text = fields.get(
                "text",
                "",
            )


            st.write(
                text[:1000] + "..."
            )


# ------------------------------------------------------------
# 13. Instructions when no question
# ------------------------------------------------------------

elif search_clicked and not query:

    st.warning(
        "Please enter a question first."
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Simple Search AI • "
    "Vespa + Hybrid Search + Reranking + "
    "RAG + Ollama/Gemma 3"
)