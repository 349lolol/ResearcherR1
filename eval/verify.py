from eval.models import Claim, CitedChunk
from eval.adapters.base import BaseModelAdapter
from typing import Generator
import re 


def extract_claims(answer: str) -> list[Claim]:
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    references = []
    for i, sentence in enumerate(sentences):
        references[i] = [int(n) for n in re.findall(r'\[(\d+)\]', sentence)]
    claims = []
    for i, points in enumerate(references):
        claims.append(Claim(
            text=sentences[i],
            cited_indices=points,
            supported=False
        ))
    return claims

def check_support(claims: list[Claim], chunks: list[CitedChunk], adapter: BaseModelAdapter)