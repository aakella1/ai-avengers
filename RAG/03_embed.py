"""STAGE 3 — VECTORIZATION (EMBEDDING)

Turn each chunk of text into a fixed-length vector of numbers. Chunks with
similar meaning land close together in that vector space — that is the whole
trick behind semantic search.

Two providers are available (see embedders.py):

    --provider openai   text-embedding-3-small, 1536 dims. Needs OPENAI_API_KEY
                        in RAG/.env. Better quality, costs ~$0.02 per million
                        tokens (this corpus is a fraction of a cent).

    --provider local    all-MiniLM-L6-v2, 384 dims. Runs on CPU, free, offline.

Whichever you choose is recorded in the metadata and reused automatically at
query time. Build one index per provider and compare them.

Run:
    python 03_embed.py                        # openai (default)
    python 03_embed.py --provider local
    python 03_embed.py --provider openai --model text-embedding-3-large
"""

import argparse

import numpy as np

from embedders import get_embedder
from rag_common import (
    CHUNKS_FILE,
    EMBED_META_FILE,
    EMBEDDINGS_FILE,
    read_json,
    write_json,
)


class Vectorizer:
    """Encodes chunk text into a matrix of shape (num_chunks, embedding_dim)."""

    def __init__(self, provider: str = "openai", model: str | None = None):
        self.embedder = get_embedder(provider, model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, show_progress=True)

    def run(self) -> np.ndarray:
        payload = read_json(CHUNKS_FILE)
        chunks = payload["chunks"]
        texts = [c["text"] for c in chunks]

        vectors = self.embed(texts)

        EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_FILE, vectors)
        write_json(
            EMBED_META_FILE,
            {
                # Both fields matter: the query must be embedded by this exact
                # provider AND model, or the vectors are not comparable.
                "provider": self.embedder.provider,
                "model": self.embedder.model_name,
                "num_vectors": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
                "normalized": True,
            },
        )
        return vectors


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into vectors.")
    parser.add_argument("--provider", default="openai", choices=["openai", "local"])
    parser.add_argument("--model", help="override the default model for the provider")
    args = parser.parse_args()

    print("STAGE 3: VECTORIZATION")
    vectorizer = Vectorizer(args.provider, args.model)
    vectors = vectorizer.run()

    print(f"\nEmbedded {vectors.shape[0]} chunks into {vectors.shape[1]}-dim vectors.")
    print(f"Saved to {EMBEDDINGS_FILE.name} ({vectors.nbytes / 1024:.0f} KB)")
    print(f"\nFirst vector (first 8 of {vectors.shape[1]} dims):")
    print(" ", np.round(vectors[0][:8], 4))
    print("\nNext: python 04_load_index.py")


if __name__ == "__main__":
    main()
