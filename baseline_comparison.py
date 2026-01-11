"""
Baseline comparison for ResearcherR1.
Naive RAG: retrieve chunks -> generate answer (no verification, no repair).
Then we use OUR verifier to grade it, to compare apples-to-apples.
"""

import json
import time
import os

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ingest.qdrant_indexer import QdrantIndexer
from eval.models import CitedChunk
from eval.verify import verify
from eval.adapters.gemini import GeminiAdapter


FLASH_MODEL = "gemini-2.5-flash"

# Simple prompt - no citation instructions, no evidence-first reasoning
NAIVE_PROMPT = """Based on the following excerpts from academic papers, answer the question.

Papers:
{context}

Question: {question}

Answer:"""


class BaselineRunner:
    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._indexer = QdrantIndexer()
        self._adapter = GeminiAdapter()  # For verification

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=60),
        retry=retry_if_exception_type((ResourceExhausted, GoogleAPIError))
    )
    def generate_answer(self, question: str) -> tuple[str, dict, list[CitedChunk]]:
        """Generate naive answer - no citation instructions."""
        # Retrieve relevant chunks using hybrid search
        results = self._indexer.hybrid_search(question, top_k=10)

        # Convert to CitedChunk for verification
        chunks = []
        for i, (doc, score) in enumerate(results):
            chunks.append(CitedChunk(
                chunk_id=str(i),
                doc_id=doc.metadata.get("doc_id", "unknown"),
                chunk_type=doc.metadata.get("chunk_type", "unknown"),
                page_start=doc.metadata.get("page_start", -1),
                page_end=doc.metadata.get("page_end", -1),
                text=doc.page_content,
                score=score
            ))

        # Build context (no indices, just raw text)
        context = "\n\n---\n\n".join([
            f"From {c.doc_id}:\n{c.text}" for c in chunks
        ])

        prompt = NAIVE_PROMPT.format(context=context, question=question)

        config = genai.types.GenerateContentConfig(
            temperature=0.2,
        )

        response = self._client.models.generate_content(
            model=FLASH_MODEL,
            contents=prompt,
            config=config,
        )

        usage = response.usage_metadata
        tokens = {
            "input": usage.prompt_token_count or 0 if usage else 0,
            "output": usage.candidates_token_count or 0 if usage else 0,
        }

        return response.text or "", tokens, chunks

    def verify_answer(self, answer: str, chunks: list[CitedChunk]):
        """Use our verification system to grade the naive answer."""
        return verify(answer, chunks, self._adapter)


def run_baseline(
    questions_path: str = "benchmark_questions.json",
    output_path: str = "baseline_results.json",
):
    with open(questions_path) as f:
        questions = json.load(f)

    runner = BaselineRunner()
    results = []
    total_start = time.time()

    print(f"Running NAIVE baseline for {len(questions)} questions...")
    print("(No citations, no EFR prompting, then verified with our system)")
    print("-" * 60)

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]

        print(f"[{i+1}/{len(questions)}] Q{qid}: {question[:50]}...")

        start = time.time()
        try:
            answer, tokens, chunks = runner.generate_answer(question)
            gen_elapsed = time.time() - start

            print(f"    Generated in {gen_elapsed:.1f}s, verifying...")

            verification, ver_gen = runner.verify_answer(answer, chunks)
            total_elapsed = time.time() - start

            result = {
                "id": qid,
                "question": question,
                "answer": answer,
                "tokens": {
                    "input": tokens["input"] + ver_gen.input_tokens,
                    "output": tokens["output"] + ver_gen.output_tokens,
                },
                "latency_s": round(total_elapsed, 2),
                "support_precision": verification.support_precision,
                "claims_total": len(verification.claims),
                "claims_supported": sum(1 for c in verification.claims if c.supported),
            }

            print(f"    -> {verification.support_precision:.1%} supported ({result['claims_supported']}/{result['claims_total']} claims)")

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
    verified = [r for r in successful if r.get("support_precision") is not None]

    total_claims = sum(r.get("claims_total", 0) for r in verified)
    total_supported = sum(r.get("claims_supported", 0) for r in verified)

    summary = {
        "total_questions": len(questions),
        "successful": len(successful),
        "failed": len(questions) - len(successful),
        "total_time_s": round(total_elapsed, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in successful) / len(successful), 2) if successful else 0,
        "avg_support_precision": round(sum(r["support_precision"] for r in verified) / len(verified), 3) if verified else 0,
        "total_claims": total_claims,
        "total_supported": total_supported,
        "overall_support_precision": round(total_supported / total_claims, 3) if total_claims else 0,
        "total_input_tokens": sum(r.get("tokens", {}).get("input", 0) for r in successful),
        "total_output_tokens": sum(r.get("tokens", {}).get("output", 0) for r in successful),
    }

    output = {
        "config": {
            "model": FLASH_MODEL,
            "method": "naive_rag",
            "description": "No EFR prompting, no citations, verified post-hoc",
            "top_k": 10,
        },
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("-" * 60)
    print("Baseline complete!")
    print(f"  Questions: {summary['successful']}/{summary['total_questions']} successful")
    print(f"  Avg support precision: {summary['avg_support_precision']:.1%}")
    print(f"  Overall: {summary['total_supported']}/{summary['total_claims']} claims supported ({summary['overall_support_precision']:.1%})")
    print(f"  Avg latency: {summary['avg_latency_s']}s")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    run_baseline()
