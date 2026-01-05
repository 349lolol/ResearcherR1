from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from ingest.qdrant_config import QdrantConfig
from ingest.langchain_embeddings import GeminiEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from ingest.qdrant_config import CORPUS_VERSION

MAX_DOCS = 100000

class QdrantIndexer:
    def __init__(self):
        self.config = QdrantConfig()
        self.client = self.config.get_client()
        self.config.create_collection(self.client)
        self.embeddings = GeminiEmbeddings()
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.config.collection_name,
            embedding=self.embeddings.get_langchain_embeddings()
        )
        self._ensure_corpus_version()
        self._bm25_retriever = None
        self._all_docs_cache = None

    
    def add_documents(self, documents: list[Document], batch_size: int = 100) -> list[str]:
        return self.vector_store.add_documents(
            documents, 
            batch_size=batch_size
        )
    
    def search(self, query: str, top_k: int = 5, filter_doc_id: str = "") -> list[tuple[Document, float]]:
        if filter_doc_id:
            return self.vector_store.similarity_search_with_score(
                query = query,
                k = top_k,
                filter = {
                    "doc_id": filter_doc_id,
                    } #type: ignore
            )
        else:
            return self.vector_store.similarity_search_with_score(
                query = query,
                k=top_k,
            )
        
    def delete_doc_by_id(self, doc_id: str) -> None:
        points_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=points_filter
        )

    def count(self) -> int:
        info = self.client.get_collection(self.config.collection_name)
        return info.points_count or 0

    def doc_exists(self, doc_id: str) -> bool:
        results = self.vector_store.similarity_search(
            query="a",
            k=1,
            filter={"doc_id": doc_id}  # type: ignore
        )
        return len(results) > 0

    def list_doc_ids(self) -> set[str]:
        doc_ids = set()
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=100,
                offset=offset,
                with_payload=["metadata.doc_id"],
                with_vectors=False
            )

            for point in points:
                if point.payload and "metadata" in point.payload:
                    doc_id = point.payload["metadata"].get("doc_id")
                    if doc_id:
                        doc_ids.add(doc_id)

            if len(doc_ids) >= MAX_DOCS:
                print(f"Total number of docs (#{len(doc_ids)})exceeded {MAX_DOCS}")
                return doc_ids

            if next_offset is None:
                break
            offset = next_offset

        return doc_ids
    
    def _ensure_corpus_version(self):
        #compare first chunk version against .env
        points, _ = self.client.scroll(
            collection_name = self.config.collection_name,
            limit = 1,
            with_payload = ["metadata.corpus_version"],
            with_vectors = False
        )

        if points and points[0].payload:
            stored_version = points[0].payload.get("metadata", {}).get("corpus_version")
            if stored_version != CORPUS_VERSION:
                print(f"Corpus version mismatch ({stored_version} != {CORPUS_VERSION}), wiping collection")
                self.client.delete_collection(self.config.collection_name)
                self.config.create_collection(self.client)

    def _load_all_documents(self) -> list[Document]:
        """Load ALL documents from Qdrant using paginated scroll."""
        if self._all_docs_cache is not None:
            return self._all_docs_cache

        all_docs = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for point in points:
                if point.payload:
                    metadata = point.payload.get("metadata", {})
                    page_content = point.payload.get("page_content", "")
                    all_docs.append(Document(
                        page_content=page_content,
                        metadata=metadata
                    ))

            if next_offset is None:
                break
            offset = next_offset

        self._all_docs_cache = all_docs
        print(f"Loaded {len(all_docs)} documents for BM25 index")
        return all_docs

    def _get_bm25_retriever(self, k: int = 10) -> BM25Retriever | None:
        """Lazy-load BM25 index from all Qdrant documents."""
        if self._bm25_retriever is None:
            all_docs = self._load_all_documents()
            if not all_docs:
                return None
            self._bm25_retriever = BM25Retriever.from_documents(all_docs)
        self._bm25_retriever.k = k
        return self._bm25_retriever

    def hybrid_search(self, query: str, top_k: int = 10, bm25_weight: float = 0.5) -> list[tuple[Document, float]]:
        """Ensemble search combining BM25 + vector via RRF."""
        bm25_retriever = self._get_bm25_retriever(k=top_k)

        # Fallback to vector-only if no documents for BM25
        if bm25_retriever is None:
            return self.search(query, top_k)

        qdrant_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[bm25_weight, 1 - bm25_weight]
        )

        docs = ensemble.invoke(query)
        # RRF doesn't produce probability scores - use 1.0 for all
        # Rank order is preserved by list position
        return [(doc, 1.0) for doc in docs[:top_k]]
