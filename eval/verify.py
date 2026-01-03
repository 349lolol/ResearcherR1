from eval.models import Claim, CitedChunk
from eval.adapters.base import BaseModelAdapter, GenerationResult

import re 
import json


BATCH_SIZE = 5

VERIFY_SYSTEM_PROMPT = """You are a claim verification assistant. For each sentence, determine if it is supported.

Rules:
- For sentences WITH citations [N]: Check if the cited evidence supports the claim. Mark supported=true only if the evidence directly supports it.
- For sentences WITHOUT citations: If structural/transitional (e.g., "Let me explain", "There are three reasons"), mark supported=true. If factual assertion without citation, mark supported=false.

Return JSON only: {"results": [{"index": 0, "supported": true}, {"index": 1, "supported": false}, ...]}"""

def extract_claims(answer: str) -> list[Claim]:
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    references = []
    for i, sentence in enumerate(sentences):
        references.append([int(n) for n in re.findall(r'\[(\d+)\]', sentence)])
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
            data = json.loads(gen.text)
            for result in data.get("results", []):
                i = result.get("index", -1)
                if 0 <= i < len(batch):
                    claims[start + i].supported = result.get("supported", False)
        except json.JSONDecodeError:
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
        if j < len(chunks):
            prompt += f"[{j}] {chunks[j].text}\n"
    return prompt