import json

from eval.models import CitedChunk, RouterPlan
from eval.adapters.base import BaseModelAdapter


def route(question: str, adapter: BaseModelAdapter) -> RouterPlan:
    system_prompt = """Generate 2-3 semantic search queries for the given question.
Return JSON: {"original_query": "<the question>", "expanded_queries": ["query1", "query2", "query3"]}
Make queries specific and varied to maximize retrieval coverage."""

    result = adapter.generate(question, system_prompt=system_prompt)
    data = json.loads(result.text)

    return RouterPlan(
        original_query=data["original_query"],
        expanded_queries=data["expanded_queries"]
    )

def build_packet(chunks: list[CitedChunk]) -> str:
    response = ""
    for i, chunk in enumerate(chunks):
        response = response + f"[{i}] {chunk.doc_id} (pages {chunk.page_start}-){chunk.page_end}, score {chunk.score})\n"
        response = response + f"{chunk.text}\n"
    return response

def deduce(question: str, evidence: str, adapter: BaseModelAdapter) -> str:
    system_prompt = """Answer the question using ONLY the evidence provided.
Cite sources using [N] format matching the evidence numbers.
Do not add information not in the evidence."""

    prompt = f"Question: {question}\n\nEvidence\n{evidence}"
    result = adapter.generate(prompt, system_prompt=system_prompt)
    return result.text
