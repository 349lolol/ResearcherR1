from google import genai
from google.genai import types
import numpy as np
from ingest.models import ChunkRecord
from ingest.models import EmbedRecord
from typing import Generator
from pathlib import Path
import orjson

client = genai.Client()

def generate_embed(chunk: ChunkRecord) -> EmbedRecord:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk.text,
        config=types.EmbedContentConfig(output_dimensionality=768, task_type="RETRIEVAL_DOCUMENT")
        )
    embeds = list(result.embeddings[0].values) if result.embeddings and result.embeddings[0].values else []
    return EmbedRecord(chunk_id=chunk.chunk_id, embedding=embeds)
        

def generate_all_embeds(chunks: list[ChunkRecord],  batch_size = 100) -> list[EmbedRecord]:
    split_chunks = chunk_splitter(chunks, batch_size)
    embeds = []

    for batch in split_chunks:
        texts = [chunk.text for chunk in batch]
        results = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=768, task_type="RETRIEVAL_DOCUMENT")
        )
        for i, chunk in enumerate(batch):
            values = list(results.embeddings[i].values) if (results.embeddings and i < len(results.embeddings) and results.embeddings[i].values) else [] #type: ignore
            embeds.append(
                EmbedRecord(
                    chunk_id=chunk.chunk_id,
                    embedding=values
                )
            )
    return embeds

def chunk_splitter(chunks: list[ChunkRecord], batch_size: int) -> Generator[list[ChunkRecord], None, None]:
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]

def _load_processed(embeds_path: Path) -> set[str]:
    if not embeds_path.exists():
        return set()
    else:
        with open(embeds_path, "rb") as f:
            return {orjson.loads(line)["chunk_id"] for line in f if line.strip()}
