import pytest
from legal_rag.generation.grounding_gate import LegalGroundingGate
from legal_rag.models import QuerySignals, RetrievalResult


@pytest.fixture
def gate():
    return LegalGroundingGate(min_relevance_floor=0.1)


def test_gate_blocks_empty_candidates(gate):
    signals = QuerySignals(
        original_query="ما هي عقوبة القتل العمد؟",
        normalized_query="ما هي عقوبة القتل العمد؟",
        language="ar"
    )
    decision = gate.evaluate(signals, candidates=[])
    assert decision.passed is False
    assert decision.refusal_reason is not None
    assert "لا تحتوي على معلومات كافية" in decision.refusal_reason


def test_gate_blocks_out_of_range_article(gate):
    signals = QuerySignals(
        original_query="ماذا تنص المادة 9999؟",
        normalized_query="ماذا تنص المادة 9999؟",
        language="ar",
        article_numbers=[9999]
    )
    # Article 9999 is outside 1..1149
    decision = gate.evaluate(signals, candidates=[])
    assert decision.passed is False
    assert "9999" in decision.refusal_reason
    assert "غير موجودة" in decision.refusal_reason


def test_gate_enforces_requested_article_presence(gate):
    signals = QuerySignals(
        original_query="ماذا تنص المادة 147؟",
        normalized_query="ماذا تنص المادة 147؟",
        language="ar",
        article_numbers=[147]
    )
    # Candidates contain only Article 200 (147 missing)
    candidate_200 = RetrievalResult(
        chunk_id="art_200",
        article_number=200,
        score=0.9,
        sources=["dense"],
        text_ar="نص المادة 200",
        text_en="Article 200 text",
        metadata={"jurisdiction": "Egypt", "law_type": "civil"}
    )
    decision = gate.evaluate(signals, candidates=[candidate_200])
    assert decision.passed is False
    assert "لم يتم العثور على أحكام المادة [147]" in decision.refusal_reason


def test_gate_passes_valid_exact_candidates(gate):
    signals = QuerySignals(
        original_query="ماذا تنص المادة 147؟",
        normalized_query="ماذا تنص المادة 147؟",
        language="ar",
        article_numbers=[147]
    )
    candidate_147 = RetrievalResult(
        chunk_id="art_147",
        article_number=147,
        score=1.0,
        sources=["exact"],
        text_ar="العقد شريعة المتعاقدين",
        text_en="The contract makes the law of the parties",
        metadata={"jurisdiction": "Egypt", "law_type": "civil"}
    )
    decision = gate.evaluate(signals, candidates=[candidate_147])
    assert decision.passed is True
    assert len(decision.final_candidates) == 1
    assert decision.final_candidates[0].article_number == 147


def test_gate_filters_scope_mismatch(gate):
    signals = QuerySignals(
        original_query="ما هي أحكام العقد في القانون المدني المصري؟",
        normalized_query="ما هي احكام العقد في القانون المدني المصري؟",
        language="ar",
        jurisdiction="Egypt",
        law_type="civil"
    )
    # Candidate with wrong jurisdiction
    saudi_candidate = RetrievalResult(
        chunk_id="art_10",
        article_number=10,
        score=0.8,
        sources=["dense"],
        text_ar="نص سعودي",
        metadata={"jurisdiction": "Saudi Arabia", "law_type": "civil"}
    )
    decision = gate.evaluate(signals, candidates=[saudi_candidate])
    assert decision.passed is False
    assert "لا توجد نصوص قانونية مطابقة لنطاق الاختصاص" in decision.refusal_reason
