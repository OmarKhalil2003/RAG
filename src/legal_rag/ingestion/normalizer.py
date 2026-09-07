import re

# Arabic-Indic to Western digits mapping
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANS_TABLE = str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)

# Arabic diacritics regex (tashkeel)
ARABIC_DIACRITICS_REGEX = re.compile(r"[\u064B-\u0652\u0653-\u065F\u0670\u06D6-\u06ED]")
# Tatweel (kashida)
TATWEEL_REGEX = re.compile(r"\u0640")
# Alef variants
ALEF_VARIANTS_REGEX = re.compile(r"[إأآا]")


def normalize_arabic_digits(text: str) -> str:
    """Normalize Arabic-Indic digits (٠-٩) to Western Arabic digits (0-9)."""
    if not text:
        return ""
    return text.translate(DIGIT_TRANS_TABLE)


def normalize_arabic_text(text: str) -> str:
    """
    Apply conservative Arabic normalization for search and indexing.
    
    Principles:
    - Removes tashkeel / diacritics.
    - Removes tatweel / elongation.
    - Normalizes Alef variants (أ, إ, آ -> ا).
    - Normalizes Arabic-Indic numerals (١ -> 1).
    - Normalizes whitespace.
    - STOPS short of conflating ة -> ه or ى -> ي to maintain legal precision.
    """
    if not text:
        return ""
    
    # 1. Normalize digits
    text = normalize_arabic_digits(text)
    
    # 2. Strip diacritics
    text = ARABIC_DIACRITICS_REGEX.sub("", text)
    
    # 3. Strip tatweel
    text = TATWEEL_REGEX.sub("", text)
    
    # 4. Harmonize Alef
    text = re.sub(r"[إأآ]", "ا", text)
    
    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def extract_integers_from_text(text: str) -> list[int]:
    """Extract all integer numbers from text after digit normalization."""
    normalized = normalize_arabic_digits(text)
    matches = re.findall(r"\b\d+\b", normalized)
    return [int(m) for m in matches]


COMMON_ARABIC_OCR_REPLACEMENTS = [
    # 1. Structural / Tatweel
    (r"[\u0640]", ""),
    # 2. Paragraph clause numbering brackets
    (r"\(\s*([٠-٩\d]+)\s*\(", r"(\1)"),
    (r"\)\s*([٠-٩\d]+)\s*\)", r"(\1)"),
    # 3. Known OCR word/phrase defects from Civil Code PDF extraction
    (r"\bنصرف أثر العقد\b", "ينصرف أثر العقد"),
    (r"\bوسد الموازنة\b", "وبعد الموازنة"),
    (r"\bالهالتين\b", "الحالتين"),
    (r"\bما لم يرف به\b", "ما لم يوف به"),
    (r"\bاعذاره\b", "إعذاره"),
    (r"\bبإعتباره\b", "باعتباره"),
    (r"\bبإلتزامه\b", "بالتزامه"),
    (r"\bاإللتزام\b", "الالتزام"),
    (r"\bإلتزاما\b", "التزاماً"),
    (r"\bإلتزام\b", "التزام"),
    (r"\bإخالل\b", "إخلال"),
    (r"\bيجهالن\b", "يجهلان"),
    (r"\bخالف\b", "خلاف"),
    (r"\bميالدي\b", "ميلادي"),
    (r"\bلإلبطال\b", "للإبطال"),
    (r"\bألهمية\b", "الأهمية"),
    # 4. Inverted Alif-Hamza and Madda with Lam
    (r"\bاأل", "الأ"),
    (r"\bاإل", "الإ"),
    (r"\bاآل", "الآ"),
    (r"\bلأل", "للأ"),
    # 5. Reverse Lam-Alef in negative particles
    (r"\bإال\b", "إلا"),
    (r"\bفال\b", "فلا"),
    (r"\bوال\b", "ولا"),
    (r"\bأال\b", "ألا"),
    (r"\bال\s+(ينصرف|يجوز|يكون|يعد|تعد|تسري|تسرى|ينفذ|تسمع|يعتد|تتناسب|يبطل|يملك|يلزم|ينتهي|تنتقل|تعتبر|يحول|يترتب|يقبل|يجبر|يسري|يصح|يقع|يرف|يوف|تبين|يقصد|يستحق|تستحق|يشمل|تشمل)\b", r"لا \1"),
    (r"\bبحيث ال\b", "بحيث لا"),
    (r"\bحتى ال\b", "حتى لا"),
    # 6. Word-end Tanween and Lam-Alef Tanween
    (r"\bباطالً?\b", "باطلاً"),
    (r"\bقابالً?\b", "قابلاً"),
    (r"\bمستحيالً?\b", "مستحيلاً"),
    (r"\bأصالً?\b", "أصلاً"),
    (r"\bمسئوالً?\b", "مسؤولاً"),
    (r"\bاستعماالً?\b", "استعمالاً"),
    (r"\bاجالً?\b", "أجلاً"),
    (r"\bمرهقا\b", "مرهقاً"),
    (r"\bحقا\b", "حقاً"),
    (r"\bمعا\b", "معاً"),
    (r"\bصحيحا\b", "صحيحاً"),
]


def clean_arabic_ocr_artifacts(text: str) -> str:
    """Repair common Arabic PDF/OCR ligature extraction errors for clean legal display."""
    if not text:
        return ""
    cleaned = text
    for pattern, repl in COMMON_ARABIC_OCR_REPLACEMENTS:
        cleaned = re.sub(pattern, repl, cleaned)
    return cleaned

