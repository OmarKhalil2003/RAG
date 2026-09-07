from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    LAWYER = "lawyer"
    CITIZEN = "citizen"
    LAW_STUDENT = "law_student"
    LEGAL_RESEARCHER = "legal_researcher"


class LegalArticle(BaseModel):
    document_id: str
    jurisdiction: str
    law_type: str
    law_name_ar: str
    law_name_en: str
    law_year: int
    article_number: int
    book: str = ""
    chapter: str = ""
    section: str = ""
    topic: str = ""
    text_ar: str
    text_en: str
    is_repealed: bool = False
    repeal_source: str = ""
    source_page: int = 1
    citation: str = ""

    def canonical_retrieval_text(self) -> str:
        """Construct canonical representation for dense and sparse indexing."""
        parts = [
            self.law_name_en,
            self.law_name_ar,
            f"Article {self.article_number}",
            f"المادة {self.article_number}"
        ]
        if self.book:
            parts.append(self.book)
        if self.chapter:
            parts.append(self.chapter)
        if self.section:
            parts.append(self.section)
        if self.topic:
            parts.append(self.topic)
        if self.text_ar:
            parts.append(self.text_ar)
        if self.text_en:
            parts.append(self.text_en)
        return "\n".join(parts)


class QuerySignals(BaseModel):
    original_query: str
    normalized_query: str
    language: str = "ar"  # 'ar', 'en', 'unknown'
    article_numbers: list[int] = Field(default_factory=list)
    jurisdiction: str | None = None
    law_type: str | None = None


class RetrievalResult(BaseModel):
    chunk_id: str
    article_number: int
    score: float
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    text_ar: str = ""
    text_en: str = ""
    is_repealed: bool = False
    citation: str = ""


class SourceCitation(BaseModel):
    document_id: str
    law_name: str
    article_number: int
    citation: str
    is_repealed: bool = False
    text_ar: str = ""
    text_en: str = ""
    hierarchy: str = ""


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    cached: bool = False
    retrieval_count: int = 0
    query_signals: QuerySignals
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
