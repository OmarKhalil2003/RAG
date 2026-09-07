import pytest
from legal_rag.cache.semantic_cache import RedisSemanticCache, cosine_distance
from legal_rag.models import RAGResponse, QuerySignals, SourceCitation, UserRole


def test_cosine_distance():
    # Identical vectors -> distance 0.0
    vec1 = [1.0, 0.0, 0.0]
    assert cosine_distance(vec1, vec1) == pytest.approx(0.0, abs=1e-5)

    # Orthogonal vectors -> distance 1.0
    vec2 = [0.0, 1.0, 0.0]
    assert cosine_distance(vec1, vec2) == pytest.approx(1.0, abs=1e-5)

    # Opposite vectors -> distance 2.0
    vec3 = [-1.0, 0.0, 0.0]
    assert cosine_distance(vec1, vec3) == pytest.approx(2.0, abs=1e-5)


def test_semantic_cache_isolation_and_hit():
    # Initialize cache in local dev mode
    cache = RedisSemanticCache(distance_threshold=0.08)

    dummy_response = RAGResponse(
        answer="العقد شريعة المتعاقدين",
        sources=[
            SourceCitation(
                document_id="egyptian_civil_code_1948",
                law_name="القانون المدني المصري",
                article_number=147,
                citation="Egyptian Civil Code, Article 147",
                is_repealed=False
            )
        ],
        cached=False,
        retrieval_count=1,
        query_signals=QuerySignals(
            original_query="ما هي آثار العقد؟",
            normalized_query="ما هي اثار العقد؟",
            language="ar"
        )
    )

    base_vector = [0.5] * 1024
    near_vector = [0.5 + 0.001] * 1024  # Extremely close (distance < 0.08)
    far_vector = [-0.5] * 1024          # Distant (distance > 0.08)

    # 1. Store response for Lawyer / Egypt / Civil
    cache.store(
        query="ما هي آثار العقد؟",
        query_vector=base_vector,
        response=dummy_response,
        jurisdiction="Egypt",
        law_type="civil",
        user_role=UserRole.LAWYER
    )

    # 2. Query with near_vector under same scope -> CACHE HIT
    hit = cache.get(
        query_vector=near_vector,
        jurisdiction="Egypt",
        law_type="civil",
        user_role=UserRole.LAWYER
    )
    assert hit is not None
    assert hit.cached is True
    assert hit.sources[0].article_number == 147

    # 3. Query with far_vector -> CACHE MISS
    miss = cache.get(
        query_vector=far_vector,
        jurisdiction="Egypt",
        law_type="civil",
        user_role=UserRole.LAWYER
    )
    assert miss is None

    # 4. Scope Isolation Check: Different Role (CITIZEN) -> CACHE MISS
    role_miss = cache.get(
        query_vector=near_vector,
        jurisdiction="Egypt",
        law_type="civil",
        user_role=UserRole.CITIZEN
    )
    assert role_miss is None

    # 5. Scope Isolation Check: Different Jurisdiction (Saudi Arabia) -> CACHE MISS
    jurisdiction_miss = cache.get(
        query_vector=near_vector,
        jurisdiction="Saudi Arabia",
        law_type="civil",
        user_role=UserRole.LAWYER
    )
    assert jurisdiction_miss is None


def test_indic_to_western_cache_hit():
    cache = RedisSemanticCache()
    dummy_response = RAGResponse(
        answer="أحكام المادة 157 من القانون المدني المصري",
        sources=[],
        cached=False,
        retrieval_count=1,
        query_signals=QuerySignals(
            original_query="ما هي أحكام المادة ١٥٧؟",
            normalized_query="ما هي احكام المادة 157؟",
            language="ar",
            article_numbers=[157]
        )
    )

    vec = [0.1] * 1024

    # 1. Store with Indic query
    cache.store(
        query="ما هي أحكام المادة ١٥٧؟",
        query_vector=vec,
        response=dummy_response,
        jurisdiction="Egypt",
        law_type="civil",
        user_role=UserRole.LAWYER,
        normalized_query="ما هي احكام المادة 157؟",
        article_numbers=[157]
    )

    # 2. Query with Western query numerals -> Must be a CACHE HIT
    hit = cache.get(
        query_vector=vec,
        jurisdiction="Egypt",
        law_type="civil",
        user_role=UserRole.LAWYER,
        normalized_query="ما هي احكام المادة 157؟",
        article_numbers=[157]
    )
    assert hit is not None
    assert hit.cached is True

