from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CitedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_type: Literal["STREAM", "PAGE"]
    page_start: int
    page_end: int
    text: str
    score: float


class RouterPlan(BaseModel):
    original_query: str
    expanded_queries: list[str]


class Claim(BaseModel):
    text: str
    cited_indices: list[int] = Field(default_factory=list)
    supported: bool = False


class VerificationResult(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    support_precision: float = 0.0


class EvalState(BaseModel):
    question: str
    router_plan: Optional[RouterPlan] = None
    retrieved_chunks: list[CitedChunk] = Field(default_factory=list)
    evidence_packet: Optional[str] = None
    draft_answer: Optional[str] = None
    verification: Optional[VerificationResult] = None
    repair_count: int = 0
    final_answer: Optional[str] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    latency_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)
