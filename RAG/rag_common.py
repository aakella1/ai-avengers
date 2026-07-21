"""Shared paths and tiny helpers for the RAG demo.

Every stage of the pipeline writes its output to disk under RAG/data/, so each
script can be run on its own and you can inspect the intermediate artifacts.
"""

import json
from pathlib import Path

RAG_DIR = Path(__file__).parent
PDF_DIR = RAG_DIR                      # the PDFs live next to the scripts
DATA_DIR = RAG_DIR / "data"

EXTRACTED_DIR = DATA_DIR / "extracted"   # stage 1 output: raw text per PDF
CHUNKS_FILE = DATA_DIR / "chunks.json"   # stage 2 output: all chunks, one list
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"   # stage 3 output: float32 matrix
EMBED_META_FILE = DATA_DIR / "embeddings_meta.json"
INDEX_ROOT = DATA_DIR / "indexes"        # stage 4 output: one folder per index

# Small, fast, runs locally on CPU. 384-dimensional vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — did you run the previous stage in the pipeline?"
        )
    return json.loads(path.read_text())


def resolve_index_id(index_id: str) -> Path:
    """Turn an index name (or the word 'latest') into a directory path."""
    if not INDEX_ROOT.exists():
        raise FileNotFoundError("No indexes yet — run 04_load_index.py first.")

    candidates = sorted(p for p in INDEX_ROOT.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError("No indexes yet — run 04_load_index.py first.")

    if index_id == "latest":
        # Sort by the recorded creation time, NOT by directory name. Index ids
        # carry a provider prefix (idx_openai_..., idx_local_...) and --name
        # can be anything, so alphabetical order is not chronological order.
        return max(candidates, key=lambda p: read_json(p / "meta.json")["created_at"])

    path = INDEX_ROOT / index_id
    if not path.is_dir():
        available = "\n  ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"Index '{index_id}' not found. Available indexes:\n  {available}"
        )
    return path
