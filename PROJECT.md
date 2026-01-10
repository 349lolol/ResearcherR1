# ResearcherR1

Evidence-First Reasoning (EFR) RAG system for academic paper Q&A with citation-grounded responses.

## Results

| Metric | EFR RAG | Raw Gemini |
|--------|---------|------------|
| Citation accuracy | 96.3% | 78% |
| Hallucination reduction | 83% | - |

Tested on 4 benchmark questions across 29 indexed papers.

## Tech Stack

- Python 3.12, LangChain, LangGraph
- Qdrant vector DB with BM25 hybrid search (RRF fusion)
- Gemini 2.5 Flash / GPT-4o
- PyMuPDF4LLM for PDF extraction

## Pipeline

```
Question → Router → Retrieve (BM25 + Vector) → Deduce → Verify → Repair
```

| Mode | Description |
|------|-------------|
| `Naive` | Generate answer, skip verification |
| `EFR` | Evidence-First Reasoning with verification |
| `EFR+Verify` | EFR + repair unsupported claims |
| `Baseline` | Standard RAG (comparison) |

## Structure

```
eval/           # Query pipeline (graph.py, verify.py, repair.py)
ingest/         # Ingestion (chunker, context_labeler, qdrant_indexer)
data/sources/   # PDF files
```

## Usage

```bash
uv sync
docker run -d -p 6333:6333 qdrant/qdrant
uv run python -m ingest.cli process-all
uv run python -m eval.cli "Your question" --model gemini --ablation EFR
```

See [USAGE.md](USAGE.md) for details.
