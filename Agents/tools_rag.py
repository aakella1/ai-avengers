"""A tool that searches the RAG corpus — agentic RAG.

This is the bridge between the two demos, and the most interesting tool in the
set to talk through in class.

In the RAG pipeline (RAG/06_generate.py) retrieval is *unconditional*: every
question runs a vector search, the chunks are pasted into the prompt, and only
then does the model see anything. The control flow is yours.

Here the exact same retriever is exposed as a tool. Now the model decides
whether to search at all, what query to search with (often not the user's
literal words), whether one search was enough, and whether to combine the
result with another tool. The control flow is the model's.

Same index, same embeddings, same FAISS call. Completely different system.

Requires an index built by the RAG pipeline first:
    cd ../RAG && python 01_extract.py && python 02_chunk.py \\
                 && python 03_embed.py && python 04_load_index.py
"""

import contextlib
import io
import os
import sys
from importlib import import_module
from pathlib import Path

from strands import tool

RAG_DIR = Path(__file__).parent.parent / "RAG"

# The RAG scripts import each other by bare name (rag_common, embedders), so
# their folder has to be importable.
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

# Which index the tool queries. "latest" follows whatever you built most
# recently; set RAG_INDEX_ID to pin a specific one (e.g. idx_openai_...).
INDEX_ID = os.environ.get("RAG_INDEX_ID", "latest")

# Loading the retriever means loading an embedding model, which is slow. Do it
# once on first use and keep it, rather than on import or on every call.
_retriever = None
_load_error: str | None = None


def set_index(index_id: str) -> None:
    """Point the tool at a different index and drop the cached retriever.

    The Streamlit UI uses this to switch between the OpenAI and local indexes
    without a restart. The next search reloads with the new embedding model.
    """
    global INDEX_ID, _retriever, _load_error
    INDEX_ID = index_id
    _retriever = None
    _load_error = None


def list_indexes() -> list[str]:
    """Index ids available on disk, newest first."""
    root = RAG_DIR / "data" / "indexes"
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if p.is_dir()), reverse=True)


def current_index() -> str:
    """The index actually loaded, or the one we will try next."""
    if _retriever is not None:
        return _retriever.index_id
    return INDEX_ID


def _get_retriever():
    """Lazily build the retriever, remembering failure so we retry only once."""
    global _retriever, _load_error

    if _retriever is not None or _load_error is not None:
        return _retriever

    try:
        # "05_retrieve" is not a valid identifier, so a normal import statement
        # cannot reach it — import_module can.
        SemanticRetriever = import_module("05_retrieve").SemanticRetriever
        # The retriever prints its provider on load; keep that out of the
        # agent's conversation output.
        with contextlib.redirect_stdout(io.StringIO()):
            _retriever = SemanticRetriever(INDEX_ID)
    except FileNotFoundError as exc:
        _load_error = (
            f"The document index is not available ({exc}). "
            "Build it by running the RAG pipeline in the RAG/ folder."
        )
    except Exception as exc:  # noqa: BLE001 - surface anything to the model as text
        _load_error = f"Could not load the document index: {type(exc).__name__}: {exc}"

    return _retriever


@tool
def search_documents(query: str, max_results: int = 4) -> str:
    """Search the private document library and return the most relevant passages.

    The library contains detailed biographies of two Indian Telugu film actors:
    Chiranjeevi and Nandamuri Balakrishna. It covers their early life and
    family, film careers and notable movies, awards and honours, political
    careers, and philanthropy.

    Use this for any question about either person. These documents are private
    and are not on the web, so web_search will not find them.

    Args:
        query: What to look for. Describe the topic in a few words — this is a
            meaning-based search, so it does not need to match the wording in
            the documents.
        max_results: How many passages to return, between 1 and 10.

    Returns:
        Numbered passages, each with its source file, page number and a
        relevance score between 0 and 1.
    """
    retriever = _get_retriever()
    if retriever is None:
        return _load_error or "The document index is unavailable."

    max_results = max(1, min(max_results, 10))

    try:
        hits = retriever.search(query, max_results)
    except Exception as exc:  # noqa: BLE001
        return f"Document search failed: {type(exc).__name__}: {exc}"

    if not hits:
        return f"No passages found for {query!r}."

    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] {hit['source']}, page {hit['page']} "
            f"(relevance {hit['score']:.3f})\n{hit['text']}"
        )

    return (
        f"Top {len(hits)} passages for {query!r} "
        f"(index: {retriever.index_id}):\n\n" + "\n\n".join(blocks)
    )
