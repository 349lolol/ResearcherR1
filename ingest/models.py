from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Source(str, Enum):
    ARXIV = "arxiv"
    MANUAL_UPLOAD = "manual_upload"
    DOI = "doi"
    URL = "url"
    LOCAL = "local"


class IngestMode(str, Enum):
    PDF_FALLBACK = "pdf_fallback"


class ChunkType(str, Enum):
    SECTION = "section"
    PAGE = "page"
    CAPTION = "caption"


class PageRecord(BaseModel):
    doc_id: str
    page: int = Field(ge=1)
    text: str

    class Config:
        extra = "forbid"


class CorpusRecord(BaseModel):
    doc_id: str = Field(min_length=1)
    source: Source
    ingest_mode: IngestMode
    file_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    processing_timestamp: datetime
    version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    pdf_path: Optional[str] = Field(None, min_length=1)
    source_dir: Optional[str] = None
    title: Optional[str] = None
    page_count: Optional[int] = Field(None, ge=1)
    arxiv_id: Optional[str] = Field(None, pattern=r"^\d{4}\.\d{4,5}(v\d+)?$")

    class Config:
        extra = "forbid"
        use_enum_values = True


class ChunkRecord(BaseModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    chunk_type: ChunkType
    text: str = Field(min_length=1)
    heading_path: Optional[str] = None
    page_start: Optional[int] = Field(None, ge=1)
    page_end: Optional[int] = Field(None, ge=1)

    class Config:
        extra = "forbid"
        use_enum_values = True
