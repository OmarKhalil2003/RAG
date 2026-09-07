import pytest
from legal_rag.pipeline import RAGService
from legal_rag.models import UserRole
from legal_rag.generation.prompts import construct_prompt
from legal_rag.generation.llm import MockLLMClient


def test_enrich_query_with_history():
    rag = RAGService(llm_client=MockLLMClient())

    # Case 1: Standalone query with article number -> untouched
    q1 = "ما هي أحكام المادة 147 من القانون المدني؟"
    res1 = rag._enrich_query_with_history(q1, history=None)
    assert res1 == q1

    # Case 2: Referential follow-up query after Article 157 inquiry
    history = [
        {"role": "user", "content": "ما هي أحكام المادة 157 من القانون المدني المصري؟"},
        {"role": "assistant", "content": "تنص المادة 157 على فسخ العقد في العقود الملزمة للجانبين..."}
    ]
    follow_up = "وما هي شروط وإجراءات ذلك الفسخ؟"
    enriched = rag._enrich_query_with_history(follow_up, history=history)
    assert "157" in enriched
    assert follow_up in enriched

    # Case 3: Follow-up that already mentions a new article -> untouched
    new_art_query = "ماذا عن المادة 148 وحسن النية؟"
    res3 = rag._enrich_query_with_history(new_art_query, history=history)
    assert res3 == new_art_query


def test_construct_prompt_with_history():
    history = [
        {"role": "user", "content": "ما هي أحكام المادة 10 من قانون التحكيم؟"},
        {"role": "assistant", "content": "يجب أن يكون اتفاق التحكيم مكتوباً وإلا كان باطلاً."}
    ]
    sys_prompt, usr_prompt = construct_prompt(
        question="وما هي الشروط الموضوعية؟",
        candidates=[],
        role=UserRole.LAWYER,
        language="ar",
        history=history
    )
    assert "PRIOR CONSULTATION CONTEXT:" in usr_prompt
    assert "اتفاق التحكيم" in usr_prompt
    assert "CURRENT USER QUESTION:" in usr_prompt


def test_ask_stream_mock_client():
    rag = RAGService(llm_client=MockLLMClient())
    resp_meta, stream_gen, finalize_fn = rag.ask_stream(
        question="ما هي أحكام المادة 147 من القانون المدني؟",
        role=UserRole.CITIZEN,
        jurisdiction="Egypt",
        law_type="civil",
        history=None
    )

    assert resp_meta.query_signals.language == "ar"
    assert resp_meta.retrieval_count > 0

    tokens = list(stream_gen)
    assert len(tokens) > 1
    assembled = "".join(tokens)

    final_resp = finalize_fn(assembled)
    assert final_resp.answer == assembled
    assert final_resp.retrieval_count == len(final_resp.sources)
    assert final_resp.latency_ms > 0
