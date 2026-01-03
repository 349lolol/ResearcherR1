from eval.models import VerificationResult
from eval.adapters.base import BaseModelAdapter, GenerationResult


REPAIR_SYSTEM_PROMPT = """You are a claim repair assistant. Revise the answer to remove or soften unsupported claims.

Rules:
- Keep all supported claims unchanged
- Remove unsupported factual claims, or soften them (e.g., "X is true" → "X may be true")
- Maintain the answer's structure and flow
- Keep all valid citations [N]

Return only the revised answer text."""


def repair(answer: str, verification: VerificationResult, evidence: str, adapter: BaseModelAdapter) -> tuple[str, GenerationResult]:
    unsupported = [c for c in verification.claims if not c.supported]

    if not unsupported:
        return answer, GenerationResult(text=answer, input_tokens=0, output_tokens=0)

    unsupported_text = "\n".join(f"- {c.text}" for c in unsupported)

    prompt = f"""Original answer:
{answer}

Unsupported claims to remove/soften:
{unsupported_text}

Evidence:
{evidence}

Revise the answer:"""

    gen = adapter.generate(prompt, system_prompt=REPAIR_SYSTEM_PROMPT)
    return gen.text, gen
