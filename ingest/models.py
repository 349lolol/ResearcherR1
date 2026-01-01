from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    """Represents a single page of extracted text from a PDF."""
    doc_id: str
    page: int = Field(ge=1)
    text: str

    class Config:
        extra = "forbid"