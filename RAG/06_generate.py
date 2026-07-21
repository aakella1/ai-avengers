"""STAGE 6 — GENERATION (the "G" in RAG)

Retrieve chunks, paste them into a prompt as context, and ask Claude to answer
using only that context.

This is the punchline of the whole demo: the model was never trained on these
specific PDFs, and we never fine-tuned anything. We just put the right text in
front of it at question time.

Use --show-prompt in class — seeing the assembled prompt is what makes RAG
click for most people.

Requires:
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python 06_generate.py --index latest --query "Compare their political careers"
    python 06_generate.py --index latest --query "..." --show-prompt
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

SemanticRetriever = import_module("05_retrieve").SemanticRetriever

# Look for the key in RAG/.env first, then Agents/.env — that way it works
# wherever you happened to put it. load_dotenv does not overwrite variables
# that are already set, so an exported key still wins.
_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / "Agents" / ".env")

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You answer questions using only the context provided by the user.

Rules:
- Use only facts present in the context. Do not add outside knowledge.
- Cite the source of each claim inline as [source, page N].
- If the context does not contain the answer, say so plainly. Do not guess."""


class RAGGenerator:
    """Retrieval + prompt assembly + one call to Claude."""

    def __init__(self, index_id: str = "latest", top_k: int = 5):
        self.retriever = SemanticRetriever(index_id)
        self.top_k = top_k
        # Reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic()

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        """Glue the retrieved chunks and the question into one user message."""
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                f"[{i}] source: {chunk['source']}, page {chunk['page']}\n"
                f"{chunk['text']}"
            )
        context = "\n\n".join(context_blocks)

        return (
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {query}"
        )

    def answer(self, query: str, show_prompt: bool = False) -> str:
        chunks = self.retriever.search(query, self.top_k)
        prompt = self.build_prompt(query, chunks)

        print(f"Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, start=1):
            print(f"  [{i}] {chunk['score']:.4f}  "
                  f"{chunk['source']} p.{chunk['page']}")

        if show_prompt:
            print("\n" + "=" * 70)
            print("PROMPT SENT TO THE MODEL")
            print("=" * 70)
            print(prompt)
            print("=" * 70)

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


def main():
    parser = argparse.ArgumentParser(description="Answer a question with RAG.")
    parser.add_argument("--index", default="latest", help="index id, or 'latest'")
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the assembled prompt before sending it")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY not found. Add it to RAG/.env or Agents/.env as:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    generator = RAGGenerator(args.index, args.top_k)
    print(f"STAGE 6: GENERATION  (index={generator.retriever.index_id}, model={MODEL})")
    print(f"Query: {args.query!r}\n")

    answer = generator.answer(args.query, show_prompt=args.show_prompt)
    print("\n--- ANSWER ---")
    print(answer)


if __name__ == "__main__":
    main()
