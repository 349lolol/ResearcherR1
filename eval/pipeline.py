import json

from eval.models import CitedChunk, RouterPlan
from eval.adapters.base import BaseModelAdapter, GenerationResult
from eval.prompts import ROUTER_SYSTEM_PROMPT, EFR_DEDUCE_SYSTEM_PROMPT, BASELINE_DEDUCE_SYSTEM_PROMPT


def _extract_json(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text


def route(question: str, adapter: BaseModelAdapter) -> tuple[RouterPlan, GenerationResult]:
    result = adapter.generate(question, system_prompt=ROUTER_SYSTEM_PROMPT)
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
    prompt = f"Question: {question}\n\nEvidence:\n{evidence}"
    result = adapter.generate(prompt, system_prompt=EFR_DEDUCE_SYSTEM_PROMPT)
    return result.text, result


def baseline_deduce(question: str, evidence: str, adapter: BaseModelAdapter) -> tuple[str, GenerationResult]:
    """Standard RAG prompt - helpful assistant with sources, no strict grounding."""
    prompt = f"Question: {question}\n\nSources:\n{evidence}"
    result = adapter.generate(prompt, system_prompt=BASELINE_DEDUCE_SYSTEM_PROMPT)
    return result.text, result
