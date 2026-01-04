from pathlib import Path

import pymupdf4llm
from tenacity import retry, stop_after_attempt, wait_exponential

from ingest.models import PageRecord


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def extract_pdf(pdf_path: Path, doc_id: str) -> list[PageRecord]:
    """Extract PDF pages as markdown using pymupdf4llm."""
    pages = []
    try:
        # Extract each page separately to preserve page boundaries
        page_docs = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,  # Return list of pages
        )

        for i, page_doc in enumerate(page_docs):
            text = page_doc.get("text", "") if isinstance(page_doc, dict) else str(page_doc)
            pages.append(PageRecord(doc_id=doc_id, page=i + 1, text=text))

    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        raise
    return pages
