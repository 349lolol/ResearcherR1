# ResearcherR1

Evidence-First Reasoning (EFR) RAG system for academic paper Q&A with citation-grounded responses. Implements a multi-stage pipeline combining query expansion, hybrid retrieval, claim verification, and automated answer repair to minimize hallucinations.

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

## Architecture

### Pipeline Flow

```
┌─────────┐    ┌────────┐    ┌──────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Question│───▶│ Router │───▶│ Retrieve │───▶│ Deduce │───▶│ Verify │───▶│ Repair │
└─────────┘    └────────┘    └──────────┘    └────────┘    └────────┘    └────────┘
                   │              │               │             │             │
              Expand to      BM25+Vector     Evidence-     Claim-by-    Fix unsup-
              2-3 queries    RRF fusion      grounded      claim check  ported claims
```

The pipeline is orchestrated via LangGraph with typed state (`EvalState`) tracking query plans, retrieved chunks, verification results, and token/cost accumulation.

### Ablation Modes

| Mode | Router | Retrieve | Deduce | Verify | Repair |
|------|--------|----------|--------|--------|--------|
| `Naive` | ✓ | ✓ | ✓ | ✗ | ✗ |
| `EFR` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `EFR+Verify` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Baseline` | ✓ | ✓ | Relaxed | ✓ | ✗ |

## Retrieval System

### Hybrid Search

Combines sparse (BM25) and dense (vector) retrieval with Reciprocal Rank Fusion:

```python
EnsembleRetriever(
    retrievers=[bm25_retriever, qdrant_retriever],
    weights=[0.5, 0.5]  # Configurable weighting
)
```

- **Vector:** Qdrant with 768-dim Gemini embeddings (COSINE distance)
- **Sparse:** BM25 via rank-bm25
- **Deduplication:** When same chunk appears from multiple queries, highest score wins

### Query Expansion

Router generates 2-3 semantically diverse queries to maximize recall:
```json
{
  "original_query": "How does RLHF work?",
  "expanded_queries": [
    "reinforcement learning from human feedback training process",
    "RLHF reward model preference learning"
  ]
}
```

## Ingestion Pipeline

```
PDF → PyMuPDF4LLM → Markdown → Clean → Chunk → Label → Index
```

### Text Cleaning

1. Header/footer removal (detects repeated text across 50%+ of pages)
2. Dehyphenation (`word-\n` → `word`)
3. Page number stripping
4. LaTeX command cleanup
5. Unicode sanitization

### Dual Chunking Strategy

**PAGE Chunks:** Full page content preserved for broader context.

**STREAM Chunks:** Semantic chunking with header awareness:
```python
# Split by Markdown headers, then recursively by character
MarkdownHeaderTextSplitter(headers=[("#","h1"), ("##","h2"), ("###","h3")])
RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=400)
```

**Quality Filtering:** Requires ≥30% alphabetic, ≥10 words, rejects table/axis-like data.

### Context Labeling

Async LLM-generated summaries prepended to each chunk:
```
[Document: arxiv_2401.12345 | Section: Methods > Training]
[Context: Describes the PPO optimization objective for reward model fine-tuning]

<chunk text>
```

## Verification System

### Claim Extraction

Parses answer into individual claims with citation indices:
```python
# "Transformers use attention [0]. They scale well [1,2]."
[
  Claim(text="Transformers use attention [0].", cited_indices=[0]),
  Claim(text="They scale well [1,2].", cited_indices=[1, 2])
]
```

### Verification Rules

| Claim Type | Rule |
|------------|------|
| Cited `[N]` | Supported if evidence at index N directly supports claim |
| Uncited factual | Always unsupported (requires citation) |
| Transitional phrases | Supported (no citation needed) |

**Metric:** `support_precision = supported_claims / total_claims`

### Answer Repair

For unsupported claims:
1. Remove claim entirely
2. Add hedging language ("may", "possibly")
3. Add citation if supporting evidence exists

## Project Structure

```
eval/
├── graph.py           # LangGraph pipeline orchestration
├── pipeline.py        # Router, deduce, evidence packet building
├── retrieve.py        # Hybrid search with deduplication
├── verify.py          # Claim extraction and batch verification
├── repair.py          # Answer repair for unsupported claims
├── models.py          # Pydantic schemas (EvalState, Claim, etc.)
└── adapters/          # Gemini, OpenAI, local LLM adapters

ingest/
├── langchain_chunker.py   # PAGE + STREAM chunking
├── context_labeler.py     # Async LLM context generation
├── qdrant_indexer.py      # Vector store + BM25 indexing
├── cleanup.py             # Text cleaning pipeline
└── extractors/pdf.py      # PyMuPDF4LLM extraction

data/sources/              # PDF files
```

## Usage

```bash
uv sync
docker run -d -p 6333:6333 qdrant/qdrant
uv run python -m ingest.cli process-all
uv run python -m eval.cli "Your question" --model gemini --ablation EFR
```

See [USAGE.md](USAGE.md) for details.
