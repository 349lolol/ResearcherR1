"""LLM-based contextual labeling for chunks."""

import asyncio
import os

from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

CONTEXT_PROMPT = """Given a chunk from an academic paper, write ONE sentence (max 25 words) describing its main topic.

Chunk:
{chunk_text}

Context:"""


class ContextLabeler:
    def __init__(self, batch_size: int = 20):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = "gemini-2.0-flash"
        self._batch_size = batch_size

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type((ResourceExhausted, GoogleAPIError))
    )
    async def _generate_context_async(self, chunk_text: str) -> str:
        """Generate context for a single chunk asynchronously."""
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=CONTEXT_PROMPT.format(chunk_text=chunk_text[:1500]),
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=40),
        )
        return response.text.strip() if response.text else ""

    async def _process_batch(self, chunks: list[Document]) -> list[str]:
        """Process a batch of chunks concurrently."""
        tasks = [self._generate_context_async(c.page_content) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Convert exceptions to empty strings
        return [r if isinstance(r, str) else "" for r in results]

    def label_chunks(self, chunks: list[Document], doc_id: str) -> list[Document]:
        """Label all chunks with context, processing in batches."""

        async def _run():
            all_contexts = []
            for i in range(0, len(chunks), self._batch_size):
                batch = chunks[i:i + self._batch_size]
                contexts = await self._process_batch(batch)
                all_contexts.extend(contexts)
                print(f"  Labeled {min(i + self._batch_size, len(chunks))}/{len(chunks)} chunks")
            return all_contexts

        contexts = asyncio.run(_run())

        # Build labeled documents
        result = []
        for chunk, context in zip(chunks, contexts):
            h1 = chunk.metadata.get("h1", "")
            h2 = chunk.metadata.get("h2", "")
            h3 = chunk.metadata.get("h3", "")
            breadcrumb = " > ".join(filter(None, [h1, h2, h3])) or "Introduction"

            prefix = f"[Document: {doc_id} | Section: {breadcrumb}]\n"
            if context:
                prefix += f"[Context: {context}]\n\n"
            else:
                prefix += "\n"

            result.append(Document(
                page_content=prefix + chunk.page_content,
                metadata=chunk.metadata
            ))

        return result
