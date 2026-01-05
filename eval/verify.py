import re
import json

from eval.models import Claim, CitedChunk, VerificationResult
from eval.adapters.base import BaseModelAdapter, GenerationResult
from eval.prompts import VERIFY_SYSTEM_PROMPT


BATCH_SIZE = 5


def _extract_json(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return text

def extract_claims(answer: str) -> list[Claim]:
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    references = []
    for i, sentence in enumerate(sentences):
        # Match both [0] [1] and [0, 1] citation formats
        # Find all bracketed content, then extract numbers from within
        brackets = re.findall(r'\[([^\]]+)\]', sentence)
        nums = []
        for bracket in brackets:
            nums.extend([int(n) for n in re.findall(r'\d+', bracket)])
        references.append(nums)
    claims = []
    for i, points in enumerate(references):
        claims.append(Claim(
            text=sentences[i],
            cited_indices=points,
            supported=False
        ))
    return claims

def check_support(claims: list[Claim], chunks: list[CitedChunk], adapter: BaseModelAdapter) -> tuple[list[Claim], GenerationResult]:
    total_in, total_out = 0, 0

    for start in range(0, len(claims), BATCH_SIZE):
        batch = claims[start:start + BATCH_SIZE]
        prompt = _build_prompt(batch, chunks)

        gen = adapter.generate(prompt, system_prompt=VERIFY_SYSTEM_PROMPT)
        total_in += gen.input_tokens
        total_out += gen.output_tokens

        try:
            clean_text = _extract_json(gen.text)
            data = json.loads(clean_text)
            for result in data.get("results", []):
                i = result.get("index", -1)
                if 0 <= i < len(batch):
                    claims[start + i].supported = result.get("supported", False)
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: batch {start}-{start + len(batch)} failed to parse JSON")

    return (claims, GenerationResult(text="", input_tokens=total_in, output_tokens=total_out))

def _build_prompt(batch: list[Claim], chunks: list[CitedChunk]) -> str:
    prompt = "Sentences to verify:\n"
    batch_indices = set()

    for i, claim in enumerate(batch):
        if claim.cited_indices:
            prompt += f"[{i}] \"{claim.text}\" (cites: {claim.cited_indices})\n"
        else:
            prompt += f"[{i}] \"{claim.text}\" (uncited)\n"
        batch_indices.update(claim.cited_indices)

    for j in sorted(batch_indices):
        if  0 <= j < len(chunks):
            prompt += f"[{j}] {chunks[j].text}\n"
    return prompt

def verify(
    answer: str,
    chunks: list[CitedChunk],
    adapter: BaseModelAdapter
) -> tuple[VerificationResult, GenerationResult]:
    claims = extract_claims(answer)
    claims, gen = check_support(claims, chunks, adapter)
    supported_count = sum (1 for c in claims if c.supported)
    precision = supported_count / len(claims) if claims else 0.0
    return (VerificationResult(claims=claims, support_precision=precision), gen)
