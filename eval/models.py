"""Pydantic models for the evaluation harness."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CitedChunk(BaseModel):
    """A retrieved chunk with citation metadata."""

    chunk_id: str
    doc_id: str
    chunk_type: Literal["STREAM", "PAGE"]
    page_start: int
    page_end: int
    text: str
    score: float


class RouterPlan(BaseModel):
    """Router output with expanded queries."""

    original_query: str
    expanded_queries: list[str]


class Claim(BaseModel):
    """An atomic claim extracted from the answer."""

    text: str
    cited_indices: list[int] = Field(default_factory=list)
    supported: bool = False


class VerificationResult(BaseModel):
    """Verification output."""

    claims: list[Claim] = Field(default_factory=list)
    support_precision: float = 0.0


class EvalState(BaseModel):
    """State flowing through the LangGraph pipeline."""

    # Input
    question: str

    # Pipeline stages
    router_plan: Optional[RouterPlan] = None
    retrieved_chunks: list[CitedChunk] = Field(default_factory=list)
    evidence_packet: Optional[str] = None
    draft_answer: Optional[str] = None
    verification: Optional[VerificationResult] = None
    repair_count: int = 0
    final_answer: Optional[str] = None

    # Metrics (flat)
    total_tokens: int = 0
    total_cost: float = 0.0
    latency_ms: float = 0.0

    # Errors
    errors: list[str] = Field(default_factory=list)
