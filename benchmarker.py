"""
Benchmark runner for ResearcherR1 EFR RAG system.
Runs all questions through the pipeline and saves results.
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from eval.config import EvalConfig
from eval.models import EvalState
from eval.graph import build_graph
from eval.adapters.gemini import GeminiAdapter
from ingest.qdrant_indexer import QdrantIndexer


def run_benchmark(
    questions_path: str = "benchmark_questions.json",
    output_path: str = "benchmark_results.json",
):
    with open(questions_path) as f:
        questions = json.load(f)

    config = EvalConfig(
        top_k=10,
        ablation="EFR+Verify",
        hybrid_search=True,
    )
    adapter = GeminiAdapter()
    indexer = QdrantIndexer()
    pipeline = build_graph(config, adapter, indexer)

    results = []
    total_start = time.time()

    print(f"Running {len(questions)} questions with EFR+Verify mode...")
    print("-" * 60)

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]

        print(f"[{i+1}/{len(questions)}] Q{qid}: {question[:60]}...")

        start = time.time()
        try:
            state = pipeline.invoke(EvalState(question=question))
            elapsed = time.time() - start

            verification = state.get("verification")
            result = {
                "id": qid,
                "question": question,
                "answer": state["final_answer"],
                "chunks_retrieved": len(state["retrieved_chunks"]),
                "queries_expanded": len(state["router_plan"].expanded_queries),
                "total_tokens": state["total_tokens"],
                "total_cost": state["total_cost"],
                "latency_s": round(elapsed, 2),
                "support_precision": verification.support_precision if verification else None,
                "claims_total": len(verification.claims) if verification else 0,
                "claims_supported": sum(1 for c in verification.claims if c.supported) if verification else 0,
                "repair_count": state.get("repair_count", 0),
                "errors": state.get("errors", []),
            }

            print(f"    -> {result['support_precision']:.1%} supported, {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            result = {
                "id": qid,
                "question": question,
                "answer": None,
                "error": str(e),
                "latency_s": round(elapsed, 2),
            }
            print(f"    -> ERROR: {e}")

        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary stats
    successful = [r for r in results if r.get("answer")]
    summary = {
        "total_questions": len(questions),
        "successful": len(successful),
        "failed": len(questions) - len(successful),
        "total_time_s": round(total_elapsed, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in successful) / len(successful), 2) if successful else 0,
        "avg_support_precision": round(sum(r["support_precision"] for r in successful if r.get("support_precision")) / len(successful), 3) if successful else 0,
        "total_tokens": sum(r.get("total_tokens", 0) for r in successful),
        "total_cost": round(sum(r.get("total_cost", 0) for r in successful), 4),
    }

    output = {
        "config": {
            "ablation": "EFR+Verify",
            "model": "gemini",
            "top_k": 10,
            "hybrid_search": True,
        },
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("-" * 60)
    print(f"Benchmark complete!")
    print(f"  Questions: {summary['successful']}/{summary['total_questions']} successful")
    print(f"  Avg support precision: {summary['avg_support_precision']:.1%}")
    print(f"  Avg latency: {summary['avg_latency_s']}s")
    print(f"  Total cost: ${summary['total_cost']:.4f}")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
