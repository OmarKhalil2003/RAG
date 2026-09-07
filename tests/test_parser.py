import json
import pytest
from pathlib import Path
from legal_rag.ingestion.parser import BilingualPDFParser
from legal_rag.ingestion.validator import CorpusValidator


def test_parser_with_processed_data():
    processed_path = Path("data/processed/egyptian_civil_code.json")
    if not processed_path.exists():
        pytest.skip("Processed dataset not yet generated.")

    with open(processed_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    # Validate article count and key ranges
    assert len(articles) == 1149

    lookup = {a["article_number"]: a for a in articles}

    # Spot-check Article 1
    art1 = lookup[1]
    assert art1["article_number"] == 1
    assert "تسرى النصوص التشريعية" in art1["text_ar"]
    assert "Provisions of laws govern" in art1["text_en"]
    assert art1["is_repealed"] is False

    # Spot-check Article 147
    art147 = lookup[147]
    assert art147["article_number"] == 147
    assert "العقد شريعة المتعاقدين" in art147["text_ar"]
    assert "contract makes the law of the parties" in art147["text_en"]
    assert art147["is_repealed"] is False

    # Spot-check Repealed Articles: 54 and 389
    art54 = lookup[54]
    assert art54["is_repealed"] is True
    assert "ملغاة" in art54["text_ar"] or "ألغيت" in art54["text_ar"]

    art389 = lookup[389]
    assert art389["is_repealed"] is True


def test_validator_detects_missing_and_repealed():
    validator = CorpusValidator(expected_range=[1, 10])
    from legal_rag.models import LegalArticle

    # Provide only 8 articles (missing 5 and 6)
    sample_articles = [
        LegalArticle(
            document_id="test",
            jurisdiction="Egypt",
            law_type="civil",
            law_name_ar="اختبار",
            law_name_en="Test",
            law_year=2024,
            article_number=i,
            text_ar=f"نص {i}",
            text_en=f"Text {i}",
            is_repealed=(i == 3)
        )
        for i in [1, 2, 3, 4, 7, 8, 9, 10]
    ]

    report = validator.validate(sample_articles)
    assert report.is_valid is False
    assert report.missing_articles == [5, 6]
    assert report.repealed_articles == [3]
    assert report.extracted_count == 8
