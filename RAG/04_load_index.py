"""STAGE 4 — LOAD INTO THE VECTOR DATABASE (FAISS)

Builds a FAISS index from the embedding matrix and saves it to its own folder.

Every run creates a NEW index with its own id, e.g. idx_20260718_142530. The
old ones are left alone, so you can build several (different chunk sizes,
different embedding models) and point the query scripts at whichever you want.

Each index folder is self-contained — it holds the vectors AND a copy of the
chunk text — so retrieval never depends on the intermediate files.

Run:
    python 04_load_index.py
    python 04_load_index.py --name small-chunks
    python 04_load_index.py --list
"""

import argparse
import shutil
from datetime import datetime

import faiss
import numpy as np

from rag_common import (
    CHUNKS_FILE,
    EMBED_META_FILE,
    EMBEDDINGS_FILE,
    INDEX_ROOT,
    read_json,
    write_json,
)


class FaissIndexLoader:
    """Creates a fresh FAISS index directory from the current embeddings."""

    def __init__(self, name: str | None = None):
        # The id is resolved in run(), once we know which embedding provider
        # produced the vectors — it goes into the default name so that
        # `--list` tells you at a glance which index is which.
        self.requested_name = name
        self.index_id: str | None = None
        self.index_dir = None

    def build(self, vectors: np.ndarray) -> faiss.Index:
        dim = vectors.shape[1]
        # IndexFlatIP = exact search by inner product. Because stage 3
        # normalized the vectors, inner product == cosine similarity.
        # "Flat" means brute force: perfectly accurate, fine up to ~1M vectors.
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    def run(self) -> str:
        vectors = np.load(EMBEDDINGS_FILE)
        embed_meta = read_json(EMBED_META_FILE)
        chunk_payload = read_json(CHUNKS_FILE)

        provider = embed_meta.get("provider", "local")
        self.index_id = (
            self.requested_name or f"idx_{provider}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        self.index_dir = INDEX_ROOT / self.index_id

        if self.index_dir.exists():
            raise SystemExit(
                f"Index '{self.index_id}' already exists at {self.index_dir}"
            )

        if vectors.shape[0] != chunk_payload["num_chunks"]:
            raise SystemExit(
                f"Mismatch: {vectors.shape[0]} vectors but "
                f"{chunk_payload['num_chunks']} chunks. Re-run 03_embed.py."
            )

        index = self.build(vectors)

        self.index_dir.mkdir(parents=True)
        faiss.write_index(index, str(self.index_dir / "index.faiss"))
        # Copy the chunks in, so this folder is everything retrieval needs.
        shutil.copy(CHUNKS_FILE, self.index_dir / "chunks.json")
        write_json(
            self.index_dir / "meta.json",
            {
                "index_id": self.index_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "num_vectors": index.ntotal,
                "dim": index.d,
                "index_type": "IndexFlatIP (cosine similarity)",
                # Recorded so retrieval can rebuild the exact same embedder.
                "embedding_provider": provider,
                "embedding_model": embed_meta["model"],
                "chunk_size": chunk_payload["chunk_size"],
                "chunk_overlap": chunk_payload["overlap"],
            },
        )
        return self.index_id


def list_indexes() -> None:
    if not INDEX_ROOT.exists() or not any(INDEX_ROOT.iterdir()):
        print("No indexes built yet.")
        return

    print(f"{'INDEX ID':<32} {'VECTORS':>7} {'DIM':>5}  {'EMBEDDING MODEL':<24} CREATED")
    for path in sorted(p for p in INDEX_ROOT.iterdir() if p.is_dir()):
        meta = read_json(path / "meta.json")
        print(
            f"{meta['index_id']:<32} {meta['num_vectors']:>7} {meta['dim']:>5}  "
            f"{meta['embedding_model']:<24} {meta['created_at']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index.")
    parser.add_argument("--name", help="custom index id (default: timestamp)")
    parser.add_argument("--list", action="store_true", help="list existing indexes")
    args = parser.parse_args()

    if args.list:
        list_indexes()
        return

    print("STAGE 4: LOADING INTO FAISS")
    index_id = FaissIndexLoader(args.name).run()

    print(f"\nCreated index: {index_id}")
    print(f"  location: data/indexes/{index_id}/")
    print("\nNext:")
    print(f'  python 05_retrieve.py --index {index_id} --query "your question"')
    print(f'  python 06_generate.py --index {index_id} --query "your question"')


if __name__ == "__main__":
    main()
