from eval.models import CitedChunk
from ingest.qdrant_indexer import QdrantIndexer

def retrieve(queries: list[str], top_k: int) -> list[CitedChunk]:
    responses = []
    indexer = QdrantIndexer()
    for query in queries:
        response = indexer.search(query, top_k)
        responses.append(response)

    deduped_chunks = {}

    for response in responses:
        for chunk, score in response:
            chunk_id = chunk.metadata["chunk_id"]
            if chunk_id not in deduped_chunks or score > deduped_chunks[chunk_id][1]:
                deduped_chunks[chunk_id] = (chunk, score)
    
    sorted_chunks = sorted(deduped_chunks.values(), key=lambda x: x[1], reverse=True)
    top_k_chunks = sorted_chunks[:top_k]
    return [
        CitedChunk(
            chunk_id=chunk.metadata["chunk_id"],
            doc_id=chunk.metadata["doc_id"],
            chunk_type=chunk.metadata["chunk_type"],
            page_start=chunk.metadata["page_start"],
            page_end=chunk.metadata["page_end"],
            text=chunk.page_content,
            score=score
        )
        for chunk, score in top_k_chunks
    ]
