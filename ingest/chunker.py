"""Dual chunking strategy: stream chunks + page chunks."""

from typing import Optional

from ingest.models import ChunkRecord, ChunkType, PageRecord


def create_stream_chunks(
    pages: list[PageRecord],
    doc_id: str,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[ChunkRecord]:
    """
    Create overlapping stream chunks for retrieval.

    Strategy:
    - Concatenate all page texts into a document stream
    - Split into ~target_tokens chunks with overlap
    - Track page_start/page_end for each chunk

    Args:
        pages: List of cleaned PageRecord objects
        doc_id: Document identifier
        target_tokens: Target chunk size in tokens (~800)
        overlap_tokens: Overlap between chunks (~100)

    Returns:
        List of ChunkRecord objects with chunk_type="section"
    """
    if not pages:
        return []

    # Concatenate all pages with boundary markers
    # Format: <PAGE_N>text</PAGE_N>
    full_text_parts = []
    page_boundaries = []  # Track character positions of page boundaries

    current_pos = 0
    for page in pages:
        marker_start = f"<PAGE_{page.page}>"
        marker_end = f"</PAGE_{page.page}>"

        page_start_pos = current_pos + len(marker_start)
        page_text = page.text
        page_end_pos = page_start_pos + len(page_text)

        full_text_parts.append(marker_start + page_text + marker_end)

        page_boundaries.append(
            {"page": page.page, "start": page_start_pos, "end": page_end_pos}
        )

        current_pos = page_end_pos + len(marker_end)

    full_text = "".join(full_text_parts)

    # Simple token estimation: split on whitespace
    # Not perfect but good enough for v1
    tokens = full_text.split()

    chunks: list[ChunkRecord] = []
    chunk_idx = 0

    i = 0
    while i < len(tokens):
        # Get chunk of ~target_tokens
        chunk_tokens = tokens[i : i + target_tokens]
        chunk_text = " ".join(chunk_tokens)

        # Find character positions in original text
        # This is approximate - we're using the tokenized version
        chunk_start_char = full_text.find(" ".join(chunk_tokens[:min(10, len(chunk_tokens))]))
        chunk_end_char = chunk_start_char + len(chunk_text)

        # Determine which pages this chunk spans
        page_start, page_end = _find_page_span(
            chunk_start_char, chunk_end_char, page_boundaries
        )

        # Create chunk record
        chunk_id = f"{doc_id}_stream_{chunk_idx:04d}"
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_type=ChunkType.SECTION,
            text=_clean_page_markers(chunk_text),
            page_start=page_start,
            page_end=page_end,
        )
        chunks.append(chunk)

        # Move forward by (target_tokens - overlap_tokens)
        chunk_idx += 1
        i += target_tokens - overlap_tokens

    return chunks


def create_page_chunks(pages: list[PageRecord], doc_id: str) -> list[ChunkRecord]:
    """
    Create one chunk per page for exact citations.

    Strategy:
    - 1:1 mapping of page → chunk
    - chunk_type="page"
    - Preserves exact page text for precise citation

    Args:
        pages: List of cleaned PageRecord objects
        doc_id: Document identifier

    Returns:
        List of ChunkRecord objects with chunk_type="page"
    """
    chunks: list[ChunkRecord] = []

    for page in pages:
        chunk_id = f"{doc_id}_page_{page.page:03d}"
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_type=ChunkType.PAGE,
            text=page.text,
            page_start=page.page,
            page_end=page.page,
        )
        chunks.append(chunk)

    return chunks


def _find_page_span(
    start_char: int, end_char: int, page_boundaries: list[dict]
) -> tuple[Optional[int], Optional[int]]:
    """
    Find which pages a character range spans.

    Args:
        start_char: Starting character position
        end_char: Ending character position
        page_boundaries: List of page boundary dicts with 'page', 'start', 'end'

    Returns:
        Tuple of (page_start, page_end) or (None, None) if not found
    """
    page_start = None
    page_end = None

    for boundary in page_boundaries:
        # Check if this chunk overlaps with this page
        if start_char <= boundary["end"] and end_char >= boundary["start"]:
            if page_start is None:
                page_start = boundary["page"]
            page_end = boundary["page"]

    return page_start, page_end


def _clean_page_markers(text: str) -> str:
    """
    Remove page markers from chunk text.

    Removes <PAGE_N> and </PAGE_N> markers.
    """
    import re

    text = re.sub(r"<PAGE_\d+>", "", text)
    text = re.sub(r"</PAGE_\d+>", "", text)
    return text.strip()


def estimate_token_count(text: str) -> int:
    """
    Simple token count estimation.

    Uses whitespace splitting as a proxy for tokens.
    Good enough for v1 - can refine later with tiktoken if needed.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text.split())
