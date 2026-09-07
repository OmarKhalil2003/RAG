import pytest
from legal_rag.ingestion.normalizer import (
    normalize_arabic_digits,
    normalize_arabic_text,
    extract_integers_from_text
)


def test_arabic_digit_normalization():
    # Test Arabic-Indic numerals to Western integers
    assert normalize_arabic_digits("١٤٧") == "147"
    assert normalize_arabic_digits("المادة ١٥٧") == "المادة 157"
    assert normalize_arabic_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"
    assert normalize_arabic_digits("Article 147") == "Article 147"


def test_conservative_arabic_normalization():
    # Tashkeel removal
    diacritics_text = "المَادَّةُ ١٤٧: العَقْدُ شَرِيعَةُ المُتَعَاقِدِينَ"
    normalized = normalize_arabic_text(diacritics_text)
    assert "َ" not in normalized
    assert "ُ" not in normalized
    assert "ِ" not in normalized
    assert "ّ" not in normalized

    # Alef harmonization (أ, إ, آ -> ا)
    alef_text = "أحكام إبرام العقد في آثاره"
    norm_alef = normalize_arabic_text(alef_text)
    assert "احكام ابرام العقد في اثاره" == norm_alef

    # Tatweel removal
    tatweel_text = "الــــعـــقــــد"
    assert normalize_arabic_text(tatweel_text) == "العقد"

    # CRITICAL: Verify conservative preservation of ة and ه (NO CONFLATION)
    taa_text = "التزامات قانونية خاصة بالمحكمة"
    norm_taa = normalize_arabic_text(taa_text)
    assert norm_taa.endswith("بالمحكمة")  # ة preserved, not converted to ه
    assert "قانونيه" not in norm_taa

    # CRITICAL: Verify preservation of ى and ي
    yaa_text = "دعوى قضائية تسري على المدين"
    norm_yaa = normalize_arabic_text(yaa_text)
    assert "دعوى" in norm_yaa  # ى preserved
    assert "تسري" in norm_yaa  # ي preserved


def test_extract_integers():
    assert extract_integers_from_text("المادة ١٤٧ والمادة 158") == [147, 158]
    assert extract_integers_from_text("Articles 54 to 80") == [54, 80]
