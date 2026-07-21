"""STAGE 5 — SEMANTIC RETRIEVAL

Embed the question with the SAME model used for the chunks, then ask FAISS for
the nearest chunk vectors. No LLM involved yet — this stage is pure search.

Worth demoing: search for "film career" and watch it return chunks that never
contain the word "film". That is the difference from keyword search.

Run:
    python 05_retrieve.py --index latest --query "Who is Chiranjeevi?"
    python 05_retrieve.py --index idx_20260718_142530 --query "political career" -k 3
"""

import argparse

import faiss

from embedders import get_embedder
from rag_common import read_json, resolve_index_id


class SemanticRetriever:
    """Loads one FAISS index and answers nearest-neighbour queries against it."""

    def __init__(self, index_id: str = "latest"):
        self.index_dir = resolve_index_id(index_id)
        self.meta = read_json(self.index_dir / "meta.json")
        self.chunks = read_json(self.index_dir / "chunks.json")["chunks"]
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))

        # Critical: the query must be embedded by the same provider AND model
        # as the chunks were, or the two sets of vectors live in unrelated
        # spaces and the search returns noise. We read both back from the
        # index metadata rather than trusting the caller to remember.
        self.embedder = get_embedder(
            self.meta.get("embedding_provider", "local"),
            self.meta["embedding_model"],
        )

    @property
    def index_id(self) -> str:
        return self.meta["index_id"]

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_vector = self.embedder.encode([query])

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, chunk_idx in zip(scores[0], indices[0]):
            if chunk_idx == -1:      # FAISS pads with -1 if k > ntotal
                continue
            chunk = self.chunks[int(chunk_idx)]
            results.append({**chunk, "score": float(score)})
        return results


def main():
    parser = argparse.ArgumentParser(description="Semantic search over an index.")
    parser.add_argument("--index", default="latest", help="index id, or 'latest'")
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    print("STAGE 5: RETRIEVAL")
    retriever = SemanticRetriever(args.index)
    print(f"  index={retriever.index_id}  {retriever.index.ntotal} vectors "
          f"x {retriever.index.d} dims")
    print(f"Query: {args.query!r}\n")

    results = retriever.search(args.query, args.top_k)
    for rank, hit in enumerate(results, start=1):
        print(f"[{rank}] score={hit['score']:.4f}  "
              f"{hit['source']} p.{hit['page']}  (chunk {hit['chunk_id']})")
        print(f"    {hit['text'][:280]}...\n")


if __name__ == "__main__":
    main()
