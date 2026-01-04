from eval.models import CitedChunk
from ingest.qdrant_indexer import QdrantIndexer

def retrieve(queries: list[str], top_k: int, indexer: QdrantIndexer) -> list[CitedChunk]:
    responses = []
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
            chunk_id=chunk.metadata.get("chunk_id", -1),
            doc_id=chunk.metadata.get("doc_id", -1),
            chunk_type=chunk.metadata.get("chunk_type", "failed"),
            page_start=chunk.metadata.get("page_start", -1),
            page_end=chunk.metadata.get("page_end", -1),
            text=chunk.page_content,
            score=score
        )
        for chunk, score in top_k_chunks
    ]
