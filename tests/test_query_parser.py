import pytest
from legal_rag.retrieval.query_parser import QueryParser


@pytest.fixture
def parser():
    return QueryParser()


def test_language_detection(parser):
    assert parser.detect_language("ما هي آثار العقد؟") == "ar"
    assert parser.detect_language("What are the effects of a contract?") == "en"
    assert parser.detect_language("المادة 147 of the civil code") == "ar"  # Primarily Arabic start
    assert parser.detect_language("Article 147") == "en"


def test_article_number_extraction_arabic(parser):
    # Western digits in Arabic text
    res = parser.extract_article_numbers("ما هي أحكام المادة 147؟")
    assert res == [147]

    # Arabic-Indic numerals
    res = parser.extract_article_numbers("ماذا تنص المادة ١٤٧؟")
    assert res == [147]

    # Multiple mentions
    res = parser.extract_article_numbers("المقارنة بين المادة 147 والمادة ١٥٨")
    assert res == [147, 158]

    # "مادة" without "ال"
    res = parser.extract_article_numbers("نص مادة 418 من القانون المدني")
    assert res == [418]


def test_article_number_extraction_english(parser):
    assert parser.extract_article_numbers("What does Article 147 provide?") == [147]
    assert parser.extract_article_numbers("Article #157 rescission rules") == [157]
    assert parser.extract_article_numbers("Provisions under art. 418") == [418]


def test_query_signals_construction(parser):
    signals = parser.parse("ما هي أحكام المادة ١٤٧؟", jurisdiction="Egypt", law_type="civil")
    assert signals.language == "ar"
    assert signals.article_numbers == [147]
    assert signals.jurisdiction == "Egypt"
    assert signals.law_type == "civil"
    assert "147" in signals.normalized_query


def test_article_range_extraction(parser):
    # English ranges
    res_en = parser.extract_article_numbers("Analyze art. 279 to 284")
    assert res_en == [279, 280, 281, 282, 283, 284]

    res_en_dash = parser.extract_article_numbers("Scope of articles 300-302")
    assert res_en_dash == [300, 301, 302]

    # Arabic ranges
    res_ar = parser.extract_article_numbers("ما حكم المواد من 275 إلى 278؟")
    assert res_ar == [275, 276, 277, 278]

