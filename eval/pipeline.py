import json

from eval.models import CitedChunk, RouterPlan
from eval.adapters.base import BaseModelAdapter, GenerationResult


def _extract_json(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text


def route(question: str, adapter: BaseModelAdapter) -> tuple[RouterPlan, GenerationResult]:
    system_prompt = """Generate 2-3 semantic search queries for the given question.
Return JSON: {"original_query": "<the question>", "expanded_queries": ["query1", "query2", "query3"]}
Make queries specific and varied to maximize retrieval coverage."""

    result = adapter.generate(question, system_prompt=system_prompt)
    try:
        clean_text = _extract_json(result.text)
        data = json.loads(clean_text)
        queries = data.get("expanded_queries", [question])
        if not queries or not isinstance(queries, list):
            queries = [question]
        return RouterPlan(
            original_query=data.get("original_query", question),
            expanded_queries=queries
        ), result
    except json.JSONDecodeError as e:
        print(f"Router JSON parse failed: {e}")
        return RouterPlan(original_query=question, expanded_queries=[question]), result

def build_packet(chunks: list[CitedChunk]) -> str:
    response = ""
    for i, chunk in enumerate(chunks):
        response += f"[{i}] {chunk.doc_id} (pages {chunk.page_start}-{chunk.page_end}, score {chunk.score:.3f})\n"
        response += f"{chunk.text}\n"
    return response

def deduce(question: str, evidence: str, adapter: BaseModelAdapter) -> tuple[str, GenerationResult]:
    system_prompt = """Answer the question using ONLY the evidence provided.
Cite sources using [N] format matching the evidence numbers.
Do not add information not in the evidence."""

    prompt = f"Question: {question}\n\nEvidence:\n{evidence}"
    result = adapter.generate(prompt, system_prompt=system_prompt)
    return result.text, result


def baseline_deduce(question: str, evidence: str, adapter: BaseModelAdapter) -> tuple[str, GenerationResult]:
    """Standard RAG prompt - helpful assistant with sources, no strict grounding."""
    system_prompt = """You are a knowledgeable research assistant. Answer questions clearly and thoroughly using the provided sources. Include citations [0], [1], etc. to reference which source supports each point."""

    prompt = f"Question: {question}\n\nSources:\n{evidence}"
    result = adapter.generate(prompt, system_prompt=system_prompt)
    return result.text, result
