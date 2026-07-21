"""Embedding providers — swap the model without touching the pipeline.

Two implementations behind one interface:

  local   all-MiniLM-L6-v2 via sentence-transformers.
          384 dims. Runs on your CPU. Free, offline, no key.

  openai  text-embedding-3-small (or -large) via the OpenAI API.
          1536 dims (3072 for -large). Costs money, needs a key, but is
          noticeably better at nuance and at non-English text.

Both return unit-normalized float32 vectors, so the rest of the pipeline —
FAISS index, cosine search, retrieval — is identical either way. That is the
point of the abstraction: the provider is a detail, not an architecture.

You cannot mix them. A query embedded by one model and an index built by the
other produce vectors in unrelated spaces, and the search returns nonsense.
That is why the provider and model name are recorded in the index metadata and
read back at query time rather than passed in again by hand.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Read RAG/.env so OPENAI_API_KEY is available without exporting it.
load_dotenv(Path(__file__).parent / ".env")

LOCAL_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "text-embedding-3-small"


class LocalEmbedder:
    """sentence-transformers, running on this machine."""

    provider = "local"

    def __init__(self, model: str = LOCAL_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        print(f"  provider=local  model={model}  (downloads ~90MB on first run)")
        self._model = SentenceTransformer(model)

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype("float32")


class OpenAIEmbedder:
    """OpenAI's hosted embedding endpoint."""

    provider = "openai"

    def __init__(self, model: str = OPENAI_MODEL):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY not found. Put it in RAG/.env as:\n"
                "  OPENAI_API_KEY=sk-...\n"
                "or run with --provider local to use the offline model."
            )

        self.model_name = model
        print(f"  provider=openai  model={model}")
        self._client = OpenAI()

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        # The API accepts many inputs per call. Batching keeps requests small
        # enough to stay well under the per-request token limit.
        batch_size = 100
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if show_progress:
                print(f"    embedding {start + len(batch)}/{len(texts)}...")
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            # Sort by index — the API does not guarantee response ordering.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))

        matrix = np.array(vectors, dtype="float32")
        # OpenAI already returns unit vectors, but normalizing again is cheap
        # and means both providers make the same guarantee to FAISS.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.clip(norms, 1e-12, None)).astype("float32")


def get_embedder(provider: str, model: str | None = None):
    """Factory: 'local' or 'openai' -> a ready embedder."""
    if provider == "local":
        return LocalEmbedder(model or LOCAL_MODEL)
    if provider == "openai":
        return OpenAIEmbedder(model or OPENAI_MODEL)
    raise SystemExit(f"Unknown provider {provider!r}. Use 'local' or 'openai'.")
