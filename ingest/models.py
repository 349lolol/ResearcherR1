"""Pydantic models for document ingestion pipeline."""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Source(str, Enum):
    """Document source types."""

    ARXIV = "arxiv"
    MANUAL_UPLOAD = "manual_upload"
    DOI = "doi"
    URL = "url"
    LOCAL = "local"


class IngestMode(str, Enum):
    """Ingestion mode types."""

    PDF_FALLBACK = "pdf_fallback"


class ChunkType(str, Enum):
    """Chunk type categories."""

    SECTION = "section"
    PAGE = "page"
    CAPTION = "caption"


class PageRecord(BaseModel):
    """Page-level data for raw page extraction and storage."""

    doc_id: str = Field(..., description="Reference to parent document")
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    text: str = Field(..., description="Full text content of the page")

    class Config:
        extra = "forbid"


class CorpusRecord(BaseModel):
    """Document-level metadata for research papers in the corpus."""

    doc_id: str = Field(
        ...,
        min_length=1,
        description="Unique document identifier (format: {source}_{identifier})",
    )
    source: Source = Field(..., description="Source system or origin of the document")
    ingest_mode: IngestMode = Field(..., description="Ingestion method used")
    file_hash: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA256 hash for integrity checking",
    )
    processing_timestamp: datetime = Field(
        ..., description="ISO 8601 timestamp when processed"
    )
    version: str = Field(
        ...,
        pattern=r"^v\d+\.\d+\.\d+$",
        description="Schema/processing version (semantic versioning)",
    )

    # Optional fields
    pdf_path: Optional[str] = Field(
        None, min_length=1, description="File system path to the PDF file"
    )
    source_dir: Optional[str] = Field(None, description="Original source directory")
    title: Optional[str] = Field(
        None, description="Document title (extracted from paper or metadata)"
    )
    page_count: Optional[int] = Field(
        None, ge=1, description="Total number of pages in the document"
    )
    arxiv_id: Optional[str] = Field(
        None,
        pattern=r"^\d{4}\.\d{4,5}(v\d+)?$",
        description="ArXiv identifier (format: YYMM.NNNNN or YYMM.NNNNNVV)",
    )

    class Config:
        extra = "forbid"
        use_enum_values = True


class ChunkRecord(BaseModel):
    """Chunk-level data for retrieval, embedding, and citation."""

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Unique chunk identifier (format: {doc_id}_chunk_{index})",
    )
    doc_id: str = Field(..., min_length=1, description="Reference to parent document")
    chunk_type: ChunkType = Field(
        ..., description="Type of chunk: section (stream), page, or caption"
    )
    text: str = Field(..., min_length=1, description="The actual text content of chunk")

    # Optional fields
    heading_path: Optional[str] = Field(
        None,
        description="Hierarchical heading path for section-based chunks (e.g., 'Introduction > Related Work')",
    )
    page_start: Optional[int] = Field(
        None, ge=1, description="Starting page number for this chunk (1-indexed)"
    )
    page_end: Optional[int] = Field(
        None, ge=1, description="Ending page number for this chunk (1-indexed)"
    )

    class Config:
        extra = "forbid"
        use_enum_values = True
