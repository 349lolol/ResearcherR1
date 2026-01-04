"""Configuration for the evaluation harness."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EvalConfig(BaseModel):
    """Evaluation settings."""

    # Ablation mode
    ablation: Literal["Naive", "EFR", "EFR+Verify", "Baseline"] = "EFR+Verify"

    # Retrieval
    top_k: int = 10

    # Verification
    max_repair_cycles: int = 1

    # Output
    trace_dir: str = "traces"
