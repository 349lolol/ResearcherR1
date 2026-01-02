import typer

from eval.config import EvalConfig
from eval.models import EvalState
from eval.graph import build_graph
from eval.adapters.openai import OpenAIAdapter
from eval.adapters.gemini import GeminiAdapter

app = typer.Typer()


def get_adapter(model: str):
    if model == "openai":
        return OpenAIAdapter()
    elif model == "gemini":
        return GeminiAdapter()
    else:
        raise ValueError(f"Unknown model: {model}")


@app.command()
def ask(
    question: str,
    model: str = typer.Option("openai", help="Model to use: openai or gemini"),
    top_k: int = typer.Option(10, help="Number of chunks to retrieve", min=1, max=50),
):
    config = EvalConfig(top_k=top_k)
    adapter = get_adapter(model)

    print(f"Question: {question}")
    print(f"Model: {model}")
    print(f"Top-k: {top_k}")
    print("-" * 40)

    pipeline = build_graph(config, adapter)
    result = pipeline.invoke(EvalState(question=question))

    print("Answer:")
    print(result["final_answer"])
    print("-" * 40)
    print(f"Chunks retrieved: {len(result['retrieved_chunks'])}")
    print(f"Queries expanded: {len(result['router_plan'].expanded_queries)}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Total cost: ${result['total_cost']:.4f}")


if __name__ == "__main__":
    app()
