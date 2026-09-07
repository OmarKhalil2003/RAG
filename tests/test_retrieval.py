import pytest
from legal_rag.models import LegalArticle, RetrievalResult
from legal_rag.retrieval.sparse import BM25SparseRetriever
from legal_rag.retrieval.hybrid import reciprocal_rank_fusion


@pytest.fixture
def sample_corpus():
    return [
        LegalArticle(
            document_id="egyptian_civil_code_1948",
            jurisdiction="Egypt",
            law_type="civil",
            law_name_ar="القانون المدني المصري",
            law_name_en="Egyptian Civil Code",
            law_year=1948,
            article_number=147,
            book="الالتزامات",
            chapter="مصادر الالتزام",
            section="العقد",
            topic="آثار العقد",
            text_ar="العقد شريعة المتعاقدين ونظرية الظروف الطارئة",
            text_en="The contract makes the law of the parties and onerous events",
            citation="Egyptian Civil Code, Article 147"
        ),
        LegalArticle(
            document_id="egyptian_civil_code_1948",
            jurisdiction="Egypt",
            law_type="civil",
            law_name_ar="القانون المدني المصري",
            law_name_en="Egyptian Civil Code",
            law_year=1948,
            article_number=157,
            book="الالتزامات",
            chapter="مصادر الالتزام",
            section="العقد",
            topic="فسخ العقد",
            text_ar="فسخ العقد لعدم وفاء المدين بالتزامه في العقود التبادلية",
            text_en="Rescission of contract for non performance in bilateral contracts",
            citation="Egyptian Civil Code, Article 157"
        )
    ]


def test_bm25_sparse_search(sample_corpus):
    retriever = BM25SparseRetriever(sample_corpus)
    # Search for "الظروف الطارئة"
    results = retriever.search("الظروف الطارئة", limit=5)
    assert len(results) > 0
    assert results[0].article_number == 147

    # Search for English "rescission"
    en_results = retriever.search("rescission", limit=5)
    assert len(en_results) > 0
    assert en_results[0].article_number == 157


def test_rrf_and_exact_priority_injection():
    # Dense results: ranked 157 first, 147 second
    dense = [
        RetrievalResult(chunk_id="157", article_number=157, score=0.9, sources=["dense"]),
        RetrievalResult(chunk_id="147", article_number=147, score=0.8, sources=["dense"])
    ]
    # Sparse results: ranked 147 first, 157 second
    sparse = [
        RetrievalResult(chunk_id="147", article_number=147, score=5.0, sources=["sparse"]),
        RetrievalResult(chunk_id="157", article_number=157, score=4.0, sources=["sparse"])
    ]
    # Exact lookup: query explicitly mentioned Article 147
    exact = [
        RetrievalResult(chunk_id="147", article_number=147, score=1.0, sources=["exact"])
    ]

    fused = reciprocal_rank_fusion(dense, sparse, exact_results=exact, limit=5)
    
    # Exact match MUST be priority 1
    assert len(fused) == 2
    assert fused[0].article_number == 147
    assert "exact" in fused[0].sources
    assert fused[1].article_number == 157
