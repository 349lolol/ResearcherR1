# ResearcherR1

An Evidence-First Reasoning (EFR) RAG system for academic paper question-answering with verifiable, citation-grounded responses.

## Purpose

Standard RAG systems retrieve context and generate answers, but often produce hallucinations or unsupported claims. ResearcherR1 addresses this with:

1. **Evidence-First Reasoning**: Generate answers strictly grounded in retrieved evidence with inline citations
2. **Claim Verification**: Automatically verify each claim is supported by cited chunks
3. **Claim Repair**: Optionally repair unsupported claims using available evidence

The goal is to outperform direct LLM queries on domain-specific corpora by providing accurate, verifiable, citation-backed answers.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Vector DB | Qdrant |
| Embeddings | Google Gemini (`text-embedding-004`) |
| LLM | Gemini 2.0 Flash, OpenAI GPT-4o |
| Framework | LangChain, LangGraph |
| PDF Extraction | PyMuPDF4LLM |
| Hybrid Search | BM25 + Vector (Reciprocal Rank Fusion) |

## Features

### Retrieval Optimizations
- **BM25 Hybrid Search**: Combines keyword (BM25) and semantic (vector) retrieval via RRF
- **Markdown Header Splitting**: Respects document structure (`#`, `##`, `###` headers)
- **Contextual Chunking**: LLM-generated context summaries + section breadcrumbs prepended to each chunk

### Pipeline Modes (Ablations)
| Mode | Description |
|------|-------------|
| `Naive` | Generate answer from evidence, skip verification |
| `EFR` | Evidence-First Reasoning with claim verification |
| `EFR+Verify` | EFR + automatic repair of unsupported claims |
| `Baseline` | Standard RAG prompt (for comparison) |

### Pipeline Flow
```
Question → Router → Retrieve → Build Packet → Deduce → Verify → Repair (optional)
                        ↓
                   [BM25 + Vector]
                   [RRF Fusion]
```

## File Structure

```
ResearcherR1/
├── data/
│   └── sources/           # PDF files to index
├── eval/                  # Evaluation & inference pipeline
│   ├── adapters/          # LLM adapters (Gemini, OpenAI, Local)
│   │   ├── base.py        # Base adapter interface
│   │   ├── gemini.py      # Gemini adapter
│   │   ├── openai.py      # OpenAI adapter
│   │   └── local.py       # Local model adapter
│   ├── cli.py             # Query CLI entry point
│   ├── config.py          # Evaluation config (top_k, ablation, etc.)
│   ├── graph.py           # LangGraph pipeline definition
│   ├── models.py          # Pydantic models (EvalState, RouterPlan, etc.)
│   ├── pipeline.py        # Core pipeline functions (route, deduce, etc.)
│   ├── prompts.py         # All LLM prompts
│   ├── repair.py          # Claim repair logic
│   ├── retrieve.py        # Hybrid retrieval (BM25 + vector)
│   └── verify.py          # Claim verification logic
├── ingest/                # Document ingestion pipeline
│   ├── cli.py             # Ingest CLI entry point
│   ├── cleanup.py         # Text cleaning utilities
│   ├── context_labeler.py # LLM contextual chunk labeling
│   ├── extractors/
│   │   └── pdf.py         # PDF extraction (PyMuPDF4LLM)
│   ├── langchain_chunker.py    # Markdown header + character splitting
│   ├── langchain_embeddings.py # Gemini embeddings wrapper
│   ├── models.py          # Pydantic models (PageRecord, etc.)
│   ├── qdrant_config.py   # Qdrant connection config
│   └── qdrant_indexer.py  # Qdrant indexing operations
├── .env                   # API keys and configuration
├── pyproject.toml         # Dependencies
├── PROJECT.md             # This file
└── USAGE.md               # Usage guide
```

## Key Files

### Ingestion
- **[ingest/cli.py](ingest/cli.py)**: Main CLI for processing PDFs (`process-all`, `process-arxiv`, `search`, etc.)
- **[ingest/langchain_chunker.py](ingest/langchain_chunker.py)**: Two-stage chunking (headers → characters)
- **[ingest/context_labeler.py](ingest/context_labeler.py)**: Async LLM labeling with batched Gemini calls
- **[ingest/qdrant_indexer.py](ingest/qdrant_indexer.py)**: Vector DB operations with hybrid search support

### Evaluation
- **[eval/cli.py](eval/cli.py)**: Query CLI entry point
- **[eval/graph.py](eval/graph.py)**: LangGraph state machine (Router → Retrieve → Deduce → Verify → Repair)
- **[eval/retrieve.py](eval/retrieve.py)**: BM25 + vector hybrid retrieval with RRF fusion
- **[eval/verify.py](eval/verify.py)**: Claim extraction and citation verification
- **[eval/repair.py](eval/repair.py)**: Unsupported claim repair
- **[eval/prompts.py](eval/prompts.py)**: All system/user prompts

## Configuration

### Environment Variables (.env)
```bash
# API Keys
GOOGLE_API_KEY=...
OPENAI_API_KEY=...

# Qdrant
QDRANT_MODE=docker
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=researcher_chunks

# Models
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.0-flash

# Corpus versioning (bump to force re-index)
CORPUS_VERSION=9
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Add PDFs and index
cp papers/*.pdf data/sources/
uv run python -m ingest.cli process-all

# 4. Query
uv run python -m eval.cli "How does PPO work in multi-agent systems?" --model gemini --ablation EFR
```

## Output Example

```
Question: How does PPO training work in multi-agent systems?
Model: gemini
Top-k: 10
Ablation: EFR
Hybrid search: True
----------------------------------------
Answer:
In multi-agent systems, PPO is utilized for collaborative training [3].
The objective function is: ℒPPO = 𝔼[min(𝑟𝑡(𝜃)𝐴𝑡, clip(...))] [3].
...
----------------------------------------
Chunks retrieved: 9
Queries expanded: 3
Total tokens: 10202
Total cost: $0.0018
Verification: 100.0% (11/11 claims)
```

## Architecture Decisions

1. **Two-stage chunking**: Split by markdown headers first, then by character for large sections. Preserves document structure while ensuring consistent chunk sizes.

2. **Contextual labeling**: Each chunk prefixed with `[Document: X | Section: A > B > C]` and LLM-generated context. Makes chunks self-contained for better retrieval and generation.

3. **BM25 + Vector hybrid**: RRF fusion of keyword and semantic search. Captures both exact matches and semantic similarity.

4. **Claim-level verification**: Extract individual claims from answers and verify each against cited evidence. Provides granular accuracy metrics.

5. **LangGraph state machine**: Clean separation of pipeline stages. Easy to add/modify nodes (e.g., add re-ranking, multi-hop retrieval).
