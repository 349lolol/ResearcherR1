# ResearcherR1 Usage Guide

## Quick Start

### 1. Setup
```bash
# Install dependencies
uv sync

# Start Qdrant (Docker required)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Configure API keys in .env
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

### 2. Index PDFs
```bash
# Add PDFs to data/sources/
cp your_papers/*.pdf data/sources/

# Process all PDFs
uv run python -m ingest.cli process-all

# Check indexing stats
uv run python -m ingest.cli stats

# List indexed documents
uv run python -m ingest.cli list-docs
```

### 3. Ask Questions
```bash
# Basic query (uses Naive mode by default)
uv run python -m eval.cli "What methods are used for X?"

# Specify model
uv run python -m eval.cli "Your question" --model gemini
uv run python -m eval.cli "Your question" --model openai

# Specify ablation mode
uv run python -m eval.cli "Your question" --ablation EFR
uv run python -m eval.cli "Your question" --ablation Baseline
uv run python -m eval.cli "Your question" --ablation Naive

# Adjust retrieval count
uv run python -m eval.cli "Your question" --top-k 15
```

## Ablation Modes

| Mode | Description |
|------|-------------|
| `Naive` | Generate answer, no verification |
| `EFR` | Evidence-First Reasoning with verification |
| `EFR+Verify` | EFR + repair unsupported claims |
| `Baseline` | Standard RAG prompt (for comparison) |

## Ingest Commands

```bash
# Process all PDFs in data/sources/
uv run python -m ingest.cli process-all

# Force reprocess (ignore already indexed)
uv run python -m ingest.cli process-all --force

# Process specific arXiv paper
uv run python -m ingest.cli process-arxiv 2401.12345

# Search indexed chunks directly
uv run python -m ingest.cli search "attention mechanism" --top-k 5

# Delete a document
uv run python -m ingest.cli delete-doc arxiv_2401.12345

# Show stats
uv run python -m ingest.cli stats
```

## Eval Commands

```bash
# Full command with all options
uv run python -m eval.cli "Your question here" \
  --model gemini \
  --ablation EFR+Verify \
  --top-k 10
```

### Output Explained
```
Question: How does X work?
Model: gemini
Top-k: 10
Ablation: EFR
Hybrid search: True
----------------------------------------
Answer:
X works by doing Y [0] and Z [3]...

Sources:
[0] arxiv_2512.23880v1, pp. 5-6
[3] arxiv_2512.24873v1, p. 12
----------------------------------------
Chunks retrieved: 10
Queries expanded: 3
Total tokens: 2500
Total cost: $0.0005
Verification: 85.7% (6/7 claims)
```

- **Sources**: Maps citation indices to document IDs and page numbers
- **Chunks retrieved**: Number of evidence chunks used
- **Queries expanded**: Router generated N search queries
- **Verification**: % of claims supported by cited evidence

## Comparing EFR vs Baseline

Run same question with different modes:
```bash
# Baseline (standard RAG)
uv run python -m eval.cli "Your question" --ablation Baseline --model gemini

# EFR (our approach)
uv run python -m eval.cli "Your question" --ablation EFR --model gemini
```

Compare the `Verification: X%` scores to see citation accuracy.

## Environment Variables

```bash
# Required
GOOGLE_API_KEY=...        # For Gemini
OPENAI_API_KEY=...        # For GPT-4

# Qdrant (defaults shown)
QDRANT_MODE=docker
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=researcher_chunks

# Models (defaults shown)
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.5-flash

# Corpus version (bump to force re-index)
CORPUS_VERSION=9
```

## Troubleshooting

### "Corpus version mismatch, wiping collection"
The CORPUS_VERSION changed. Re-index your PDFs:
```bash
uv run python -m ingest.cli process-all
```

### "Missing key inputs argument"
API key not loaded. Check your `.env` file exists and has valid keys.

### "Chunks retrieved: 0"
No documents indexed. Run:
```bash
uv run python -m ingest.cli stats
uv run python -m ingest.cli process-all
```

### Low verification scores
- Check if your question matches corpus content
- Try broader questions about topics in your papers
- Use `ingest.cli search` to see what chunks exist
