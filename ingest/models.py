from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    doc_id: str
    page: int = Field(ge=1)
    text: str

    class Config:
        extra = "forbid"