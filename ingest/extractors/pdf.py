"""PDF text extraction using pypdf."""

from pathlib import Path

from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

from ingest.models import PageRecord


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def extract_pdf(pdf_path: Path, doc_id: str) -> list[PageRecord]:
    """
    Extract text page-by-page from PDF using pypdf.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier

    Returns:
        List of PageRecord objects, one per page

    Raises:
        Exception: If PDF extraction fails after retries
    """
    pages: list[PageRecord] = []

    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        for page_num in range(total_pages):
            page = reader.pages[page_num]
            text = page.extract_text()

            # Create page record (1-indexed)
            page_record = PageRecord(
                doc_id=doc_id, page=page_num + 1, text=text if text else ""
            )

            # Flag low-density pages (likely extraction failure or scanned image)
            if len(text.strip()) < 50:
                print(
                    f"⚠️  Warning: {doc_id} page {page_num + 1} has <50 chars (likely scan or extraction failure)"
                )

            pages.append(page_record)

    except Exception as e:
        print(f"❌ Error extracting PDF {pdf_path}: {e}")
        raise

    return pages
