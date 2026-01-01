from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ingest.models import PageRecord

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
                    "page_end": page.page
                }
            )
            for page in pages
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
            result.append(
                Document(
                    page_content = chunk.page_content,
                    metadata={
                        "chunk_id": f"{doc_id}_stream_{i:04d}",
                        "doc_id": doc_id,
                        "chunk_type": "STREAM",
                        "page_start": page_start,
                        "page_end": page_end
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