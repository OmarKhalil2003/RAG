import pytest
from legal_rag.models import UserRole, RetrievalResult
from legal_rag.generation.prompts import construct_prompt
from legal_rag.generation.llm import MockLLMClient


@pytest.fixture
def candidates():
    return [
        RetrievalResult(
            chunk_id="art_147",
            article_number=147,
            score=1.0,
            sources=["exact", "dense"],
            text_ar="العقد شريعة المتعاقدين ، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين",
            text_en="The contract makes the law of the parties",
            citation="Egyptian Civil Code, Article 147",
            is_repealed=False
        )
    ]


def test_mock_llm_generation_lawyer_role(candidates):
    mock_llm = MockLLMClient()
    sys_p, usr_p = construct_prompt(
        question="ما هي آثار العقد؟",
        candidates=candidates,
        role=UserRole.LAWYER,
        language="ar"
    )
    answer = mock_llm.generate(system_prompt=sys_p, user_prompt=usr_p)
    assert "المادة 147" in answer or "147" in answer
    assert "العقد شريعة المتعاقدين" in answer
    assert "القانون المدني المصري" in answer


def test_mock_llm_generation_citizen_role(candidates):
    mock_llm = MockLLMClient()
    sys_p, usr_p = construct_prompt(
        question="ما هي آثار العقد؟",
        candidates=candidates,
        role=UserRole.CITIZEN,
        language="ar"
    )
    answer = mock_llm.generate(system_prompt=sys_p, user_prompt=usr_p)
    assert "المادة 147" in answer or "147" in answer
    assert "ببساطة" in answer or "العقد شريعة المتعاقدين" in answer


def test_mock_llm_generation_repealed_notice():
    mock_llm = MockLLMClient()
    repealed_cand = [
        RetrievalResult(
            chunk_id="art_54",
            article_number=54,
            score=1.0,
            sources=["exact"],
            text_ar="ألغيت المواد من 54 إلى 80",
            text_en="Articles 54-80 have been repealed",
            citation="Egyptian Civil Code, Article 54 (Repealed)",
            is_repealed=True
        )
    ]
    sys_p, usr_p = construct_prompt(
        question="ما هو الوضع القانوني للمادة 54؟",
        candidates=repealed_cand,
        role=UserRole.LAWYER,
        language="ar"
    )
    answer = mock_llm.generate(system_prompt=sys_p, user_prompt=usr_p)
    assert "54" in answer
    assert "إلغاؤها" in answer or "ملغاة" in answer or "repealed" in answer.lower()


def test_gemini_client_error_on_missing_key():
    from legal_rag.generation.llm import GeminiLLMClient
    client = GeminiLLMClient(api_key=None)
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        client.generate(system_prompt="sys", user_prompt="usr")


def test_get_llm_client_factory():
    from legal_rag.generation.llm import get_llm_client, MockLLMClient, GeminiLLMClient
    # Mock mode fallback
    client = get_llm_client(provider="mock")
    assert isinstance(client, MockLLMClient)

    # Gemini mode with key
    gemini_client = get_llm_client(provider="gemini", api_key="test-key-xyz", model="gemini-2.5-flash")
    assert isinstance(gemini_client, GeminiLLMClient)
    assert gemini_client.api_key == "test-key-xyz"
    assert gemini_client.model_name == "gemini-2.5-flash"

