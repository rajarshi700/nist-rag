from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class SourceDocument:
    id: str
    title: str
    filename: str
    url: str
    publication_url: str
    first_body_page: int = 1
    last_body_page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    document_id: str
    document: str
    pdf_page: int
    printed_page: str | None
    section: str | None
    url: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None


class StructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class QueryPlan(StructuredResponse):
    query: str = Field(min_length=1, max_length=700)
    depends_on_history: bool


class Assessment(StructuredResponse):
    verdict: Literal["sufficient", "insufficient", "out_of_scope"]
    relevant_ids: list[str] = Field(max_length=10)
    reason: str = Field(max_length=500)
    suggested_query: str = Field(max_length=700)


class Claim(StructuredResponse):
    text: str = Field(min_length=1, max_length=1200)
    source_ids: list[str] = Field(min_length=1, max_length=6)


class GroundedAnswer(StructuredResponse):
    answerable: bool
    claims: list[Claim] = Field(max_length=8)


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[dict]
    query: str
    trace: list[str]
    status: str
