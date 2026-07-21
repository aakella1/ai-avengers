"""STAGE 2 — CHUNKING

Split the extracted text into overlapping windows of words.

Why chunk at all? Two reasons worth saying out loud in class:
  1. Embedding models have a small input limit (~256-512 tokens here).
  2. Retrieval precision. If you embed a whole 24-page document as one vector,
     that vector is an average of everything and matches nothing well.

Why overlap? So a sentence that straddles a chunk boundary still appears whole
in at least one chunk.

Run:
    python 02_chunk.py
    python 02_chunk.py --chunk-size 200 --overlap 50
"""

import argparse

from rag_common import CHUNKS_FILE, EXTRACTED_DIR, read_json, write_json


class Chunker:
    """Fixed-size sliding window over words, with overlap between windows."""

    def __init__(self, chunk_size: int = 180, overlap: int = 40):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[str]:
        words = text.split()
        step = self.chunk_size - self.overlap

        chunks = []
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(" ".join(window))
            # Stop once the window has reached the end of the document,
            # otherwise the last few chunks would be near-duplicates.
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def chunk_document(self, doc: dict) -> list[dict]:
        """One document -> a list of chunk records carrying their metadata."""
        records = []
        for page in doc["pages"]:
            for chunk_text in self.split_text(page["text"]):
                records.append(
                    {
                        "text": chunk_text,
                        "source": doc["source"],
                        "page": page["page"],
                    }
                )
        return records

    def run(self) -> list[dict]:
        files = sorted(EXTRACTED_DIR.glob("*.json"))
        if not files:
            raise SystemExit("No extracted text found — run 01_extract.py first.")

        all_chunks = []
        for path in files:
            doc = read_json(path)
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  {doc['source']:28} -> {len(chunks):4} chunks")

        # Give every chunk a stable id. This id is what the vector index will
        # return, and how we map a search hit back to its text.
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_id"] = i

        write_json(
            CHUNKS_FILE,
            {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "num_chunks": len(all_chunks),
                "chunks": all_chunks,
            },
        )
        return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted text.")
    parser.add_argument("--chunk-size", type=int, default=180, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="words of overlap")
    args = parser.parse_args()

    print(f"STAGE 2: CHUNKING (size={args.chunk_size} words, overlap={args.overlap})")
    chunks = Chunker(args.chunk_size, args.overlap).run()

    print(f"\n{len(chunks)} chunks written to {CHUNKS_FILE.name}")
    print("\n--- sample chunk ---")
    print(chunks[0]["text"][:300], "...")
    print("\nNext: python 03_embed.py")


if __name__ == "__main__":
    main()
