from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EvalConfig(BaseModel):
    ablation: Literal["Naive", "EFR", "EFR+Verify", "Baseline"] = "EFR+Verify"
    top_k: int = 10
    hybrid_search: bool = True
    max_repair_cycles: int = 1
    trace_dir: str = "traces"
