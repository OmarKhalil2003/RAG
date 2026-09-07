import re
from legal_rag.models import QuerySignals
from legal_rag.ingestion.normalizer import normalize_arabic_digits, normalize_arabic_text

# Regex patterns for detecting article references in Arabic and English
AR_ARTICLE_QUERY_RE = re.compile(
    r'(?:المادة|مادة|الماده|ماده)\s*([٠-٩\d]+)',
    re.UNICODE | re.IGNORECASE
)
EN_ARTICLE_QUERY_RE = re.compile(
    r'\b(?:article|art\.?)\s*#?\s*(\d+)\b',
    re.IGNORECASE
)

# Regex patterns for detecting article ranges (e.g. "المواد من 279 إلى 302", "art. 279 to 302")
AR_ARTICLE_RANGE_RE = re.compile(
    r'(?:المواد|المادتين|مادتين|مواد)\s*(?:من\s*)?([٠-٩\d]+)\s*(?:إلى|الى|وحتى|-)\s*([٠-٩\d]+)',
    re.UNICODE | re.IGNORECASE
)
EN_ARTICLE_RANGE_RE = re.compile(
    r'\b(?:articles?|arts?\.?)\s*#?\s*(\d+)\s*(?:to|through|-)\s*#?\s*(\d+)\b',
    re.IGNORECASE
)

class QueryParser:
    """
    Analyzes legal queries to extract language, explicit article numbers, and scope.
    Supports single article references and explicit article range queries (e.g. 279 to 302).
    """

    def parse(
        self,
        query: str,
        jurisdiction: str | None = None,
        law_type: str | None = None
    ) -> QuerySignals:
        query_clean = query.strip()
        
        # 1. Detect language
        language = self.detect_language(query_clean)
        
        # 2. Extract article numbers deterministically (single or ranges)
        article_numbers = self.extract_article_numbers(query_clean)
        
        # 3. Normalize query text for embedding and search
        normalized_query = normalize_arabic_text(query_clean)
        
        return QuerySignals(
            original_query=query_clean,
            normalized_query=normalized_query.strip(),
            language=language,
            article_numbers=article_numbers,
            jurisdiction=jurisdiction,
            law_type=law_type
        )

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect whether the query is primarily Arabic or English."""
        ar_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        en_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        # If there are Arabic characters present, treat as Arabic query
        if ar_chars > 0:
            return "ar"
        elif en_chars > 0:
            return "en"
        return "ar"

    @staticmethod
    def extract_article_numbers(text: str) -> list[int]:
        """Extract referenced article numbers and article ranges from Arabic or English expressions."""
        found: set[int] = set()

        # Check Arabic ranges (e.g. "المواد من 279 إلى 302")
        for m in AR_ARTICLE_RANGE_RE.finditer(text):
            s = int(normalize_arabic_digits(m.group(1)))
            e = int(normalize_arabic_digits(m.group(2)))
            if s > e:
                s, e = e, s
            if 0 < (e - s) <= 35:
                found.update(range(s, e + 1))

        # Check English ranges (e.g. "art. 279 to 302", "articles 279-302")
        for m in EN_ARTICLE_RANGE_RE.finditer(text):
            s = int(normalize_arabic_digits(m.group(1)))
            e = int(normalize_arabic_digits(m.group(2)))
            if s > e:
                s, e = e, s
            if 0 < (e - s) <= 35:
                found.update(range(s, e + 1))

        # Check Arabic single mentions
        for m in AR_ARTICLE_QUERY_RE.finditer(text):
            raw_num = m.group(1)
            norm = normalize_arabic_digits(raw_num)
            if norm.isdigit():
                found.add(int(norm))

        # Check English single mentions
        for m in EN_ARTICLE_QUERY_RE.finditer(text):
            raw_num = m.group(1)
            norm = normalize_arabic_digits(raw_num)
            if norm.isdigit():
                found.add(int(norm))

        return sorted(list(found))
