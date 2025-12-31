from google import genai
from google.genai import types
import numpy as np
from ingest.models import ChunkRecord
from ingest.models import EmbedRecord

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
    