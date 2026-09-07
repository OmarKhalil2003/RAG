import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Plus

from legal_rag.models import LegalArticle, RetrievalResult
from legal_rag.ingestion.normalizer import normalize_arabic_text


def tokenize_legal_text(text: str) -> list[str]:
    """Tokenize legal text for BM25 lexical matching."""
    norm = normalize_arabic_text(text).lower()
    tokens = re.findall(r'\b\w+\b', norm, re.UNICODE)
    return tokens


class BM25SparseRetriever:
    """
    BM25 lexical retriever over canonical legal retrieval representations.
    Provides sparse lexical matching for legal terms, article references, and phrasing.
    """

    def __init__(self, articles: list[LegalArticle] | None = None):
        self.articles = articles or []
        self.corpus_tokens: list[list[str]] = []
        self.bm25: BM25Plus | None = None

        if self.articles:
            self._build_index()

    def _build_index(self) -> None:
        self.corpus_tokens = [
            tokenize_legal_text(a.canonical_retrieval_text())
            for a in self.articles
        ]
        self.bm25 = BM25Plus(self.corpus_tokens)

    def save(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"articles": self.articles, "corpus_tokens": self.corpus_tokens}, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "BM25SparseRetriever":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        retriever = cls(articles=data["articles"])
        retriever.corpus_tokens = data["corpus_tokens"]
        retriever.bm25 = BM25Plus(retriever.corpus_tokens)
        return retriever

    def search(
        self,
        query: str,
        limit: int = 20,
        jurisdiction: str | None = None,
        law_type: str | None = None
    ) -> list[RetrievalResult]:
        if not self.bm25 or not self.articles:
            return []

        query_tokens = tokenize_legal_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            art = self.articles[idx]
            # Apply metadata filters
            if jurisdiction and art.jurisdiction != jurisdiction:
                continue
            if law_type and art.law_type != law_type:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=f"{art.document_id}_{art.article_number}",
                    article_number=art.article_number,
                    score=float(score),
                    sources=["sparse"],
                    metadata=art.model_dump(),
                    text_ar=art.text_ar,
                    text_en=art.text_en,
                    is_repealed=art.is_repealed,
                    citation=art.citation
                )
            )
            if len(results) >= limit:
                break

        return results
