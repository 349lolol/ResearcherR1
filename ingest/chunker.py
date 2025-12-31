import re

from ingest.models import ChunkRecord, ChunkType, PageRecord


def create_stream_chunks(
    pages: list[PageRecord], doc_id: str, target_tokens: int = 400, overlap: int = 100
) -> list[ChunkRecord]:
    if not pages:
        return []

    # Build full text with page markers
    full_text = ""
    boundaries = []
    pos = 0

    for page in pages:
        marker_start = f"<PAGE_{page.page}>"
        marker_end = f"</PAGE_{page.page}>"
        start_pos = pos + len(marker_start)
        end_pos = start_pos + len(page.text)

        full_text += marker_start + page.text + marker_end
        boundaries.append({"page": page.page, "start": start_pos, "end": end_pos})
        pos = end_pos + len(marker_end)

    # Split into token chunks
    tokens = full_text.split()
    chunks = []
    i = 0

    while i < len(tokens):
        chunk_tokens = tokens[i : i + target_tokens]
        chunk_text = " ".join(chunk_tokens)

        # Find page span
        start_char = full_text.find(" ".join(chunk_tokens[:min(10, len(chunk_tokens))]))
        end_char = start_char + len(chunk_text)
        page_start, page_end = _find_page_span(start_char, end_char, boundaries)

        # Clean markers and create chunk
        clean_text = re.sub(r"</?PAGE_\d+>", "", chunk_text).strip()
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc_id}_stream_{len(chunks):04d}",
                doc_id=doc_id,
                chunk_type=ChunkType.SECTION,
                text=clean_text,
                page_start=page_start,
                page_end=page_end,
            )
        )

        i += target_tokens - overlap

    return chunks


def create_page_chunks(pages: list[PageRecord], doc_id: str) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=f"{doc_id}_page_{page.page:03d}",
            doc_id=doc_id,
            chunk_type=ChunkType.PAGE,
            text=page.text,
            page_start=page.page,
            page_end=page.page,
        )
        for page in pages
    ]


def _find_page_span(start_char: int, end_char: int, boundaries: list[dict]):
    page_start = page_end = None
    for b in boundaries:
        if start_char <= b["end"] and end_char >= b["start"]:
            if page_start is None:
                page_start = b["page"]
            page_end = b["page"]
    return page_start, page_end
