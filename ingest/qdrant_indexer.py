from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from ingest.qdrant_config import QdrantConfig
from ingest.langchain_embeddings import GeminiEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue


class QdrantIndexer:
    """Direct interface to Qdrant. No JSONL, no intermediate storage."""

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
                filter = {"doc_id": filter_doc_id} #type: ignore
            )
        else:
            return self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
        
    def delete_doc_by_id(self, doc_id: str) -> None:
        points_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=points_filter
        )