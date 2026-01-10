import warnings
from pathlib import Path

import pymupdf4llm
from tenacity import retry, stop_after_attempt, wait_exponential

from ingest.models import PageRecord

# Suppress pymupdf4llm's suggestion to use pymupdf_layout
warnings.filterwarnings("ignore", message=".*pymupdf_layout.*")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def extract_pdf(pdf_path: Path, doc_id: str) -> list[PageRecord]:
    pages = []
    try:
        page_docs = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            ignore_images=True,
            ignore_graphics=True,
        )

        for i, page_doc in enumerate(page_docs):
            text = page_doc.get("text", "") if isinstance(page_doc, dict) else str(page_doc)
            pages.append(PageRecord(doc_id=doc_id, page=i + 1, text=text))

    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        raise
    return pages
