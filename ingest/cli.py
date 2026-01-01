import re
from pathlib import Path

import typer

from ingest.cleanup import clean_pages
from ingest.extractors.pdf import extract_pdf
from ingest.langchain_chunker import LangChainChunker
from ingest.qdrant_indexer import QdrantIndexer

app = typer.Typer()


@app.command()
def process_all(
    source_dir: Path = Path("data/sources"),
    force: bool = False
):
    """Process all PDFs and index to Qdrant."""
    indexer = QdrantIndexer()
    chunker = LangChainChunker()

    processed = set() if force else indexer.list_doc_ids()
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
    for source, doc_id in to_process:
        if _process_doc(source, doc_id, chunker, indexer):
            successes += 1
        else:
            failures += 1

    print(f"Processed {len(to_process)} documents: {successes} ok, {failures} failed")


@app.command()
def process_arxiv(arxiv_id: str, force: bool = False):
    """Process a specific arXiv paper."""
    doc_id = f"arxiv_{arxiv_id}"
    indexer = QdrantIndexer()

    if not force and indexer.doc_exists(doc_id):
        print(f"{doc_id} already processed (use --force to reprocess)")
        return

    source_dir = Path("data/sources")
    pdf_candidates = list(source_dir.glob(f"*{arxiv_id}*.pdf"))

    if not pdf_candidates:
        print(f"Could not find PDF for {arxiv_id}")
        return

    chunker = LangChainChunker()
    _process_doc(pdf_candidates[0], doc_id, chunker, indexer)


@app.command()
def search(query: str, top_k: int = 5, doc_id: str = ""):
    """Search indexed documents."""
    indexer = QdrantIndexer()
    results = indexer.search(query, top_k=top_k, filter_doc_id=doc_id)

    for doc, score in results:
        print(f"\n[{score:.3f}] {doc.metadata['chunk_id']}")
        print(f"Pages {doc.metadata['page_start']}-{doc.metadata['page_end']}")
        print(doc.page_content[:200] + "...")


@app.command()
def stats():
    """Show index statistics."""
    indexer = QdrantIndexer()
    count = indexer.count()
    doc_ids = indexer.list_doc_ids()
    print(f"{len(doc_ids)} documents, {count:,} chunks")


@app.command()
def delete_doc(doc_id: str):
    """Delete a document."""
    indexer = QdrantIndexer()
    indexer.delete_doc_by_id(doc_id)
    print(f"Deleted {doc_id}")


@app.command()
def list_docs():
    """List all documents."""
    indexer = QdrantIndexer()
    doc_ids = sorted(indexer.list_doc_ids())

    if not doc_ids:
        print("No documents indexed")
        return

    print(f"{len(doc_ids)} documents:")
    for doc_id in doc_ids:
        print(f"  {doc_id}")


def _process_doc(
    source_path: Path,
    doc_id: str,
    chunker: LangChainChunker,
    indexer: QdrantIndexer
) -> bool:
    try:
        if not (source_path.is_file() and source_path.suffix == ".pdf"):
            print(f"Not a PDF: {source_path}")
            return False

        pages = extract_pdf(source_path, doc_id)
        if not pages:
            print(f"No pages extracted from {source_path}")
            return False

        cleaned_pages, _ = clean_pages(pages)

        stream_chunks = chunker.create_stream_chunks(cleaned_pages, doc_id)
        page_chunks = chunker.create_page_chunks(cleaned_pages, doc_id)
        all_chunks = stream_chunks + page_chunks

        chunk_ids = indexer.add_documents(all_chunks)

        print(f"Indexed {doc_id}: {len(chunk_ids)} chunks")
        return True

    except Exception as e:
        error = str(e)
        if "RetryError" in error:
            import traceback
            for line in traceback.format_exc().split("\n"):
                if "Exception:" in line or "Error:" in line:
                    error = line.strip()
                    break
        print(f"Failed {doc_id}: {error}")
        return False


def _extract_arxiv_id(filename: str):
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", filename)
    return match.group(1) if match else None


if __name__ == "__main__":
    app()
