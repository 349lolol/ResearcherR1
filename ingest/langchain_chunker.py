import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ingest.models import PageRecord

from ingest.qdrant_config import CORPUS_VERSION


def _is_low_quality(text: str, min_alpha_ratio: float = 0.3, min_words: int = 10) -> bool:
    """Check if chunk is mostly numbers/noise (figure axes, tables, etc.)."""
    if not text.strip():
        return True

    # Count alphabetic vs total characters
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text.replace(" ", "").replace("\n", ""))

    if total_chars == 0:
        return True

    alpha_ratio = alpha_chars / total_chars

    # Count words (sequences of letters)
    words = re.findall(r'[a-zA-Z]{2,}', text)

    # Check for figure axis patterns (sequences of numbers)
    number_sequences = re.findall(r'(?:\d+\s+){3,}', text)  # 3+ numbers in a row
    has_axis_pattern = len(number_sequences) > 0

    # Check for repeated short lines (common in figure text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    short_lines = sum(1 for l in lines if len(l) < 30)
    mostly_short_lines = len(lines) > 3 and short_lines / len(lines) > 0.6

    return (alpha_ratio < min_alpha_ratio or
            len(words) < min_words or
            has_axis_pattern or
            mostly_short_lines)

class LangChainChunker:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size * 4,
            chunk_overlap = chunk_overlap * 4,
            separators = ["\n\n", "\n", ". ", " ", ""],
            add_start_index = True
        )

    def create_page_chunks(self, pages: list[PageRecord], doc_id: str) -> list[Document]:
        if not pages:
            return []

        return [
            Document(
                page_content=page.text,
                metadata={
                    "chunk_id": f"{doc_id}_page_{page.page:03d}",
                    "doc_id": doc_id,
                    "chunk_type": "PAGE",
                    "page_start": page.page,
                    "page_end": page.page,
                    "corpus_version": CORPUS_VERSION
                }
            )
            for page in pages
            if not _is_low_quality(page.text)
        ]

    def create_stream_chunks(self, pages: list[PageRecord], doc_id: str) -> list[Document]:
        if not pages:
            return []
        
        full_text = "\n\n".join(page.text for page in pages)
        chunks = self.splitter.create_documents([full_text])
        boundaries = self._compute_page_boundaries(pages)

        result = []
        for i, chunk in enumerate(chunks):
            chunk_start = chunk.metadata.get('start_index', 0)
            chunk_end = chunk_start + len(chunk.page_content)

            page_start, page_end = self._find_page_span(chunk_start, chunk_end, boundaries)
            if page_start == -1 or page_end == -1:
                print(f"chunk {doc_id}_stream_{i:04d} has corrupted boundaries")
            elif _is_low_quality(chunk.page_content):
                pass  # Skip low-quality chunks (figure axes, tables, etc.)
            else:
                result.append(
                    Document(
                        page_content = chunk.page_content,
                        metadata={
                            "chunk_id": f"{doc_id}_stream_{i:04d}",
                            "doc_id": doc_id,
                            "chunk_type": "STREAM",
                            "page_start": page_start,
                            "page_end": page_end,
                            "corpus_version": CORPUS_VERSION
                        }
                    )
                )
        return result

    def _compute_page_boundaries(self, pages: list[PageRecord]) -> list[dict]:
        if not pages:
            return []
        boundaries = []
        pos = 0

        for page in pages:
            start = pos
            end = pos + len(page.text)

            boundaries.append({
                "page": page.page,
                "start": start,
                "end": end
            })

            pos = end + 2
        return boundaries

    def _find_page_span(self, chunk_start: int, chunk_end: int, boundaries: list[dict]) -> tuple[int, int]:
        start_page = -1
        end_page = -1

        for boundary in boundaries:
            if chunk_start <= boundary["end"] and chunk_end >= boundary["start"]:
                if start_page == -1:
                    start_page = boundary["page"]
                end_page = boundary["page"]

        return (start_page, end_page)