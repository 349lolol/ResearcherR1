"""CLI interface for document ingestion pipeline."""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ingest.chunker import create_page_chunks, create_stream_chunks
from ingest.cleanup import clean_pages
from ingest.extractors.pdf import extract_pdf
from ingest.models import CorpusRecord, IngestMode, PageRecord, Source

app = typer.Typer()
console = Console()

# Schema version for all processed documents
SCHEMA_VERSION = "v1.0.0"


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_processed_docs(corpus_path: Path) -> set[str]:
    """Load set of already-processed doc_ids from corpus.jsonl."""
    processed = set()

    if not corpus_path.exists():
        return processed

    with open(corpus_path, "rb") as f:
        for line in f:
            if line.strip():
                record = orjson.loads(line)
                processed.add(record["doc_id"])

    return processed


def _extract_arxiv_id(filename: str) -> Optional[str]:
    """
    Extract arXiv ID from filename.

    Examples:
    - "2512.22605v1.pdf" → "2512.22605v1"
    - "arXiv-2512.22605v1" → "2512.22605v1"
    """
    import re

    # Try pattern: YYMM.NNNNNvV
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", filename)
    if match:
        return match.group(1)
    return None


def _process_document(
    source_path: Path,
    doc_id: str,
    force: bool = False,
) -> Optional[tuple[CorpusRecord, list[PageRecord], list, list]]:
    """
    Process a single document through the pipeline.

    Returns:
        Tuple of (corpus_record, pages, stream_chunks, page_chunks) or None if failed
    """
    try:
        # Step 1: Extract pages from PDF
        if not (source_path.is_file() and source_path.suffix == ".pdf"):
            console.print(f"Error: Only PDF files are supported: {source_path}")
            return None

        pages = extract_pdf(source_path, doc_id)

        # Use resolve() to handle absolute vs relative paths
        try:
            pdf_path = str(source_path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            # If relative_to fails, use absolute path
            pdf_path = str(source_path.resolve())

        if not pages:
            console.print(f"Error: No pages extracted from {source_path}")
            return None

        # Step 2: Clean pages
        cleaned_pages, cleanup_stats = clean_pages(pages)

        # Step 3: Create chunks (both types)
        stream_chunks = create_stream_chunks(cleaned_pages, doc_id)
        page_chunks = create_page_chunks(cleaned_pages, doc_id)

        # Step 4: Create corpus record
        file_hash = _compute_file_hash(source_path)
        arxiv_id = _extract_arxiv_id(source_path.name)

        corpus_record = CorpusRecord(
            doc_id=doc_id,
            source=Source.ARXIV,
            ingest_mode=IngestMode.PDF_FALLBACK,
            file_hash=file_hash,
            processing_timestamp=datetime.now(),
            version=SCHEMA_VERSION,
            pdf_path=pdf_path,
            source_dir=None,
            page_count=len(pages),
            arxiv_id=arxiv_id,
        )

        console.print(
            f"Successfully processed {doc_id}, {len(stream_chunks)} stream chunks, {len(page_chunks)} page chunks"
        )
        return corpus_record, cleaned_pages, stream_chunks, page_chunks

    except Exception as e:
        # Try to extract underlying error from RetryError
        error_msg = str(e)
        if "RetryError" in error_msg:
            import traceback
            tb = traceback.format_exc()
            # Extract the actual error from traceback
            lines = tb.split('\n')
            for i, line in enumerate(lines):
                if 'Exception:' in line or 'Error:' in line:
                    error_msg = lines[i].strip()
                    break
        console.print(f"Failed to process {doc_id}: {error_msg}")
        return None


@app.command()
def process_all(
    source_dir: Path = typer.Option(
        Path("data/sources"), help="Directory containing source documents"
    ),
    force: bool = typer.Option(False, "--force", help="Reprocess already-processed documents"),
):
    """Process all unprocessed documents in source directory."""
    # Setup paths
    corpus_path = Path("data/corpus/corpus.jsonl")
    pages_path = Path("data/pages/pages.jsonl")
    chunks_path = Path("data/chunks/chunks.jsonl")

    # Create output directories
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pages_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-processed docs
    processed_docs = _load_processed_docs(corpus_path) if not force else set()

    # Discover all PDF sources
    pdf_files = list(source_dir.glob("*.pdf"))
    all_sources = pdf_files

    # Filter out processed
    sources_to_process = []
    for source in all_sources:
        doc_id = f"arxiv_{_extract_arxiv_id(source.name) or source.stem}"
        if doc_id not in processed_docs:
            sources_to_process.append((source, doc_id))

    if not sources_to_process:
        console.print("All documents already processed")
        return

    # Process each document
    successes = 0
    failures = 0

    with open(corpus_path, "ab") as corpus_f, open(pages_path, "ab") as pages_f, open(
        chunks_path, "ab"
    ) as chunks_f:
        for source, doc_id in sources_to_process:
            result = _process_document(source, doc_id, force)

            if result:
                corpus_record, pages, stream_chunks, page_chunks = result

                # Write to JSONL files
                corpus_f.write(orjson.dumps(corpus_record.model_dump()) + b"\n")

                for page in pages:
                    pages_f.write(orjson.dumps(page.model_dump()) + b"\n")

                for chunk in stream_chunks + page_chunks:
                    chunks_f.write(orjson.dumps(chunk.model_dump()) + b"\n")

                successes += 1
            else:
                failures += 1

    # Summary
    console.print(f"\nProcessed {len(sources_to_process)} documents: {successes} successful, {failures} failed")


@app.command()
def process_arxiv(
    arxiv_id: str,
    force: bool = typer.Option(False, "--force", help="Reprocess if already processed"),
):
    """Process a specific arXiv paper by ID (e.g., '2512.22605v1')."""
    doc_id = f"arxiv_{arxiv_id}"

    # Setup paths
    corpus_path = Path("data/corpus/corpus.jsonl")
    pages_path = Path("data/pages/pages.jsonl")
    chunks_path = Path("data/chunks/chunks.jsonl")

    # Create output directories
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pages_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if not force:
        processed_docs = _load_processed_docs(corpus_path)
        if doc_id in processed_docs:
            console.print(f"Document {doc_id} already processed. Use --force to reprocess.")
            return

    # Find PDF source file
    source_dir = Path("data/sources")
    pdf_candidates = list(source_dir.glob(f"*{arxiv_id}*.pdf"))

    if not pdf_candidates:
        console.print(f"Error: Could not find PDF for arXiv:{arxiv_id} in {source_dir}")
        return

    source_path = pdf_candidates[0]

    # Process document
    result = _process_document(source_path, doc_id, force)

    if result:
        corpus_record, pages, stream_chunks, page_chunks = result

        # Write to JSONL files
        with open(corpus_path, "ab") as corpus_f, open(pages_path, "ab") as pages_f, open(
            chunks_path, "ab"
        ) as chunks_f:
            corpus_f.write(orjson.dumps(corpus_record.model_dump()) + b"\n")

            for page in pages:
                pages_f.write(orjson.dumps(page.model_dump()) + b"\n")

            for chunk in stream_chunks + page_chunks:
                chunks_f.write(orjson.dumps(chunk.model_dump()) + b"\n")


if __name__ == "__main__":
    app()
