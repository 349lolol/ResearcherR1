import hashlib
import re
from datetime import datetime
from pathlib import Path

import orjson
import typer

from ingest.chunker import create_page_chunks, create_stream_chunks
from ingest.cleanup import clean_pages
from ingest.extractors.pdf import extract_pdf
from ingest.models import CorpusRecord, IngestMode, Source

app = typer.Typer()
SCHEMA_VERSION = "v1.0.0"


def process_all(
    source_dir: Path = Path("data/sources"), force: bool = False
):
    corpus_path = Path("data/corpus/corpus.jsonl")
    pages_path = Path("data/pages/pages.jsonl")
    chunks_path = Path("data/chunks/chunks.jsonl")

    for p in [corpus_path, pages_path, chunks_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    processed = _load_processed(corpus_path) if not force else set()
    pdf_files = list(source_dir.glob("*.pdf"))
    to_process = [
        (f, f"arxiv_{_extract_arxiv_id(f.name) or f.stem}")
        for f in pdf_files
        if f"arxiv_{_extract_arxiv_id(f.name) or f.stem}" not in processed
    ]

    if not to_process:
        print("All documents already processed")
        return

    successes = failures = 0
    with open(corpus_path, "ab") as cf, open(pages_path, "ab") as pf, open(
        chunks_path, "ab"
    ) as chf:
        for source, doc_id in to_process:
            result = _process_doc(source, doc_id)
            if result:
                corpus, pages, stream, page = result
                cf.write(orjson.dumps(corpus.model_dump()) + b"\n")
                for p in pages:
                    pf.write(orjson.dumps(p.model_dump()) + b"\n")
                for c in stream + page:
                    chf.write(orjson.dumps(c.model_dump()) + b"\n")
                successes += 1
            else:
                failures += 1

    print(f"Processed {len(to_process)} documents: {successes} successful, {failures} failed")


def process_arxiv(arxiv_id: str, force: bool = False):
    doc_id = f"arxiv_{arxiv_id}"
    corpus_path = Path("data/corpus/corpus.jsonl")
    pages_path = Path("data/pages/pages.jsonl")
    chunks_path = Path("data/chunks/chunks.jsonl")

    for p in [corpus_path, pages_path, chunks_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    if not force and doc_id in _load_processed(corpus_path):
        print(f"Document {doc_id} already processed. Use --force to reprocess.")
        return

    source_dir = Path("data/sources")
    pdf_candidates = list(source_dir.glob(f"*{arxiv_id}*.pdf"))

    if not pdf_candidates:
        print(f"Error: Could not find PDF for arXiv:{arxiv_id}")
        return

    result = _process_doc(pdf_candidates[0], doc_id)
    if result:
        corpus, pages, stream, page = result
        with open(corpus_path, "ab") as cf, open(pages_path, "ab") as pf, open(
            chunks_path, "ab"
        ) as chf:
            cf.write(orjson.dumps(corpus.model_dump()) + b"\n")
            for p in pages:
                pf.write(orjson.dumps(p.model_dump()) + b"\n")
            for c in stream + page:
                chf.write(orjson.dumps(c.model_dump()) + b"\n")


def _process_doc(source_path: Path, doc_id: str):
    try:
        if not (source_path.is_file() and source_path.suffix == ".pdf"):
            print(f"Error: Only PDF files supported: {source_path}")
            return None

        pages = extract_pdf(source_path, doc_id)
        if not pages:
            print(f"Error: No pages extracted from {source_path}")
            return None

        try:
            pdf_path = str(source_path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            pdf_path = str(source_path.resolve())

        cleaned_pages, _ = clean_pages(pages)
        stream_chunks = create_stream_chunks(cleaned_pages, doc_id)
        page_chunks = create_page_chunks(cleaned_pages, doc_id)

        sha256 = hashlib.sha256()
        with open(source_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        corpus = CorpusRecord(
            doc_id=doc_id,
            source=Source.ARXIV,
            ingest_mode=IngestMode.PDF_FALLBACK,
            file_hash=sha256.hexdigest(),
            processing_timestamp=datetime.now(),
            version=SCHEMA_VERSION,
            pdf_path=pdf_path,
            source_dir=None,
            page_count=len(pages),
            arxiv_id=_extract_arxiv_id(source_path.name),
        )

        print(
            f"Successfully processed {doc_id}, {len(stream_chunks)} stream chunks, {len(page_chunks)} page chunks"
        )
        return corpus, cleaned_pages, stream_chunks, page_chunks

    except Exception as e:
        error = str(e)
        if "RetryError" in error:
            import traceback

            for line in traceback.format_exc().split("\n"):
                if "Exception:" in line or "Error:" in line:
                    error = line.strip()
                    break
        print(f"Failed to process {doc_id}: {error}")
        return None


def _load_processed(corpus_path: Path) -> set[str]:
    if not corpus_path.exists():
        return set()
    with open(corpus_path, "rb") as f:
        return {orjson.loads(line)["doc_id"] for line in f if line.strip()}


def _extract_arxiv_id(filename: str):
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", filename)
    return match.group(1) if match else None


app.command()(process_all)
app.command()(process_arxiv)

if __name__ == "__main__":
    app()
