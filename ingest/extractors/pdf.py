from pathlib import Path

from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

from ingest.models import PageRecord


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def extract_pdf(pdf_path: Path, doc_id: str) -> list[PageRecord]:
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            pages.append(PageRecord(doc_id=doc_id, page=page_num + 1, text=text or ""))
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        raise
    return pages
