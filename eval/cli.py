import re

import typer
from dotenv import load_dotenv
load_dotenv()

from eval.config import EvalConfig
from eval.models import EvalState, CitedChunk
from eval.graph import build_graph
from eval.adapters.openai import OpenAIAdapter
from eval.adapters.gemini import GeminiAdapter
from eval.adapters.local import LocalAdapter
from ingest.qdrant_indexer import QdrantIndexer

app = typer.Typer()


def _extract_location(chunk: CitedChunk) -> str:
    """Extract location info from chunk - prefer page numbers (academic standard)."""
    # Primary: page numbers (academic standard)
    if chunk.page_start >= 0:
        if chunk.page_start == chunk.page_end:
            return f"p. {chunk.page_start}"
        return f"pp. {chunk.page_start}-{chunk.page_end}"

    # Fallback: section breadcrumb from contextual labeling
    match = re.search(r'\[Section: ([^\]]+)\]', chunk.text)
    if match:
        return match.group(1)

    return ""


def build_citation_legend(chunks: list[CitedChunk], answer: str) -> str:
    """Build a legend mapping citation indices to document sources."""
    # Extract which indices are actually cited in the answer
    # Handle both [0] and [0, 4] formats
    brackets = re.findall(r'\[([^\]]+)\]', answer)
    cited_indices = set()
    for bracket in brackets:
        cited_indices.update(int(n) for n in re.findall(r'\d+', bracket))

    if not cited_indices:
        return ""

    lines = ["", "Sources:"]
    for i in sorted(cited_indices):
        if 0 <= i < len(chunks):
            chunk = chunks[i]
            location = _extract_location(chunk)
            if location:
                lines.append(f"[{i}] {chunk.doc_id}, {location}")
            else:
                lines.append(f"[{i}] {chunk.doc_id}")

    return "\n".join(lines)


def get_adapter(model: str):
    if model == "openai":
        return OpenAIAdapter()
    elif model == "gemini":
        return GeminiAdapter()
    elif model == "local":
        return LocalAdapter()
    else:
        raise ValueError(f"Unknown model: {model}")


@app.command()
def ask(
    question: str,
    model: str = typer.Option("openai", help="Model: openai, gemini, or local"),
    top_k: int = typer.Option(10, help="Number of chunks to retrieve", min=1, max=50),
    ablation: str = typer.Option("Naive", help="Mode: Naive, EFR, or EFR+Verify"),
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Enable BM25 hybrid search"),
):
    config = EvalConfig(top_k=top_k, ablation=ablation, hybrid_search=hybrid)
    adapter = get_adapter(model)
    indexer = QdrantIndexer()

    print(f"Question: {question}")
    print(f"Model: {model}")
    print(f"Top-k: {top_k}")
    print(f"Ablation: {ablation}")
    print(f"Hybrid search: {hybrid}")
    print("-" * 40)

    pipeline = build_graph(config, adapter, indexer)
    result = pipeline.invoke(EvalState(question=question))

    print("Answer:")
    print(result["final_answer"])

    # Print citation legend
    legend = build_citation_legend(result["retrieved_chunks"], result["final_answer"])
    if legend:
        print(legend)

    print("-" * 40)
    print(f"Chunks retrieved: {len(result['retrieved_chunks'])}")
    print(f"Queries expanded: {len(result['router_plan'].expanded_queries)}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Total cost: ${result['total_cost']:.4f}")

    if result.get("verification"):
        v = result["verification"]
        supported = sum(1 for c in v.claims if c.supported)
        print(f"Verification: {v.support_precision:.1%} ({supported}/{len(v.claims)} claims)")
    if result.get("repair_count"):
        print(f"Repaired: {result['repair_count']} cycle(s)")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")


if __name__ == "__main__":
    app()
