"""STAGE 1 — EXTRACTION

Turn PDFs into plain text. Nothing clever happens here, but it is where most
real-world RAG pipelines quietly go wrong: if extraction is bad, everything
downstream is bad.

Run:
    python 01_extract.py
    python 01_extract.py --pdf-dir /some/other/folder
"""

import argparse
from pathlib import Path

from pypdf import PdfReader

from rag_common import EXTRACTED_DIR, PDF_DIR, write_json


class PDFExtractor:
    """Reads every PDF in a folder and writes out its text, page by page."""

    def __init__(self, pdf_dir: Path = PDF_DIR, out_dir: Path = EXTRACTED_DIR):
        self.pdf_dir = Path(pdf_dir)
        self.out_dir = Path(out_dir)

    def find_pdfs(self) -> list[Path]:
        return sorted(self.pdf_dir.glob("*.pdf"))

    def extract_one(self, pdf_path: Path) -> dict:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            # extract_text() returns None for image-only pages.
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": page_number, "text": text})

        return {
            "source": pdf_path.name,
            "num_pages": len(reader.pages),
            "num_pages_with_text": len(pages),
            "pages": pages,
        }

    def run(self) -> list[dict]:
        pdfs = self.find_pdfs()
        if not pdfs:
            raise SystemExit(f"No PDFs found in {self.pdf_dir}")

        documents = []
        for pdf_path in pdfs:
            doc = self.extract_one(pdf_path)
            out_path = self.out_dir / f"{pdf_path.stem}.json"
            write_json(out_path, doc)
            documents.append(doc)

            chars = sum(len(p["text"]) for p in doc["pages"])
            print(
                f"  {doc['source']:28} "
                f"{doc['num_pages_with_text']}/{doc['num_pages']} pages with text, "
                f"{chars:,} chars -> {out_path.relative_to(self.out_dir.parent.parent)}"
            )
        return documents


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs.")
    parser.add_argument("--pdf-dir", default=PDF_DIR, type=Path)
    args = parser.parse_args()

    print("STAGE 1: EXTRACTION")
    docs = PDFExtractor(pdf_dir=args.pdf_dir).run()
    print(f"\nExtracted {len(docs)} document(s). Next: python 02_chunk.py")


if __name__ == "__main__":
    main()
