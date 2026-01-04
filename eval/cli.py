import typer

from eval.config import EvalConfig
from eval.models import EvalState
from eval.graph import build_graph
from eval.adapters.openai import OpenAIAdapter
from eval.adapters.gemini import GeminiAdapter
from eval.adapters.local import LocalAdapter
from ingest.qdrant_indexer import QdrantIndexer

app = typer.Typer()


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
):
    config = EvalConfig(top_k=top_k, ablation=ablation)
    adapter = get_adapter(model)
    indexer = QdrantIndexer()

    print(f"Question: {question}")
    print(f"Model: {model}")
    print(f"Top-k: {top_k}")
    print(f"Ablation: {ablation}")
    print("-" * 40)

    pipeline = build_graph(config, adapter, indexer)
    result = pipeline.invoke(EvalState(question=question))

    print("Answer:")
    print(result["final_answer"])
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
