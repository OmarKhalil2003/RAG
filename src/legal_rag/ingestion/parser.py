import re
import json
from pathlib import Path
import fitz  # PyMuPDF
from legal_rag.models import LegalArticle
from legal_rag.ingestion.normalizer import normalize_arabic_digits, normalize_arabic_text

# Regular expressions for Article identification
EN_ART_RE = re.compile(r'(?:^|\n)\s*(?:Article|rticle)\s*(\d+)\b', re.IGNORECASE)
AR_ART_RE = re.compile(r'(?:^|\n)\s*(?:\(\s*مادة|مادة)\s*([٠-٩\d]+)\s*(?:\)|\b)', re.UNICODE)

# Hierarchy patterns
BOOK_RE = re.compile(r'(?:الكتاب\s+[^\n]+|BOOK\s+[^\n]+)', re.IGNORECASE)
CHAPTER_RE = re.compile(r'(?:الباب\s+[^\n]+|CHAPTER\s+[^\n]+)', re.IGNORECASE)
SECTION_RE = re.compile(r'(?:الفصل\s+[^\n]+|SECTION\s+[^\n]+|Section\s+[^\n]+)', re.IGNORECASE)

# Repeal notice pattern: e.g. "Articles 54-80 have been repealed" or "Articles 389-417 repealed"
EN_REPEAL_RE = re.compile(r'Articles?\s*(\d+)\s*-\s*(\d+)\s+(?:have been repealed|repealed)', re.IGNORECASE)
AR_REPEAL_RE = re.compile(r'ألغيت\s+المواد\s+من\s*([٠-٩\d]+)\s*إلى\s*([٠-٩\d]+)', re.UNICODE)


def format_hierarchy_block(text: str) -> str:
    """Format a spanning hierarchy block into a bilingual clean heading."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    ar_lines = []
    en_lines = []
    for l in lines:
        if any('\u0600' <= c <= '\u06FF' for c in l):
            ar_lines.append(l)
        else:
            en_lines.append(l)
    ar_text = " - ".join(ar_lines)
    en_text = " - ".join(en_lines)
    ar_text = re.sub(r'الثان\s*-\s*ي', 'الثاني', ar_text)
    ar_text = normalize_arabic_text(ar_text)
    if ar_text and en_text:
        return f"{ar_text} | {en_text}"
    return ar_text or en_text


class BilingualPDFParser:
    """
    Positional PyMuPDF parser for bilingual legal codes (Arabic / English).
    Separates two-column text, extracts article units, hierarchy, and repeal notices.
    """

    def __init__(self, config_path: str | Path | dict | None = None):
        if config_path is None:
            config_path = Path("data/corpus_config.json")
        
        if isinstance(config_path, (str, Path)):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = config_path

        self.document_id = self.config.get("document_id", "egyptian_civil_code_1948")
        self.jurisdiction = self.config.get("jurisdiction", "Egypt")
        self.law_type = self.config.get("law_type", "civil")
        self.law_name_ar = self.config.get("law_name_ar", "القانون المدني المصري")
        self.law_name_en = self.config.get("law_name_en", "Egyptian Civil Code")
        self.law_year = self.config.get("law_year", 1948)
        self.expected_range = self.config.get("expected_article_range", [1, 1149])

    def parse(self, pdf_path: str | Path) -> list[LegalArticle]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        doc = fitz.open(pdf_path)
        articles: dict[int, LegalArticle] = {}

        current_book = ""
        current_chapter = ""
        current_section = ""
        current_topic = ""

        # First pass: parse each page
        for p_idx in range(len(doc)):
            page = doc[p_idx]
            page_num = p_idx + 1
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))  # Sort top to bottom

            left_lines = []
            right_lines = []

            for b in blocks:
                x0, y0, x1, y1, text, _, _ = b
                t = text.strip()
                if not t:
                    continue

                # Check for spanning hierarchy header blocks across both columns (width > 380)
                is_book = bool(re.search(r'(?:^|\n)\s*(?:الكتاب\s+|BOOK\s+)', t, re.I))
                is_chapter = bool(re.search(r'(?:^|\n)\s*(?:الباب\s+|CHAPTER\s+)', t, re.I))
                is_section = bool(re.search(r'(?:^|\n)\s*(?:الفصل\s+|SECTION\s+|Section\s+)', t, re.I))

                if (is_book or is_chapter or is_section) and (x1 - x0 > 380):
                    formatted_heading = format_hierarchy_block(t)
                    if is_book:
                        current_book = formatted_heading
                    if is_chapter:
                        current_chapter = formatted_heading
                    if is_section:
                        current_section = formatted_heading
                    continue

                # Fallback inline regex update for non-spanning blocks
                b_match = BOOK_RE.search(t)
                if b_match:
                    current_book = b_match.group(0).strip()
                c_match = CHAPTER_RE.search(t)
                if c_match:
                    current_chapter = c_match.group(0).strip()
                s_match = SECTION_RE.search(t)
                if s_match:
                    current_section = s_match.group(0).strip()

                # Separate columns by coordinate
                if x0 < 290 and x1 < 310:
                    left_lines.append(t)
                elif x0 >= 285:
                    right_lines.append(t)
                else:
                    # Line-by-line language separation for spanning blocks
                    for line in t.split("\n"):
                        l_s = line.strip()
                        if not l_s:
                            continue
                        if any('\u0600' <= char <= '\u06FF' for char in l_s):
                            right_lines.append(l_s)
                        else:
                            left_lines.append(l_s)

            left_text = "\n".join(left_lines)
            right_text = "\n".join(right_lines)
            full_page_text = page.get_text("text")

            # Check for repeal notices on this page
            self._extract_repeal_notices(
                full_page_text, page_num, current_book, current_chapter, current_section, articles
            )

            # Split English articles
            en_splits = EN_ART_RE.split(left_text)
            page_en_articles = {}
            if len(en_splits) > 1:
                for i in range(1, len(en_splits), 2):
                    art_num = int(en_splits[i])
                    body = en_splits[i + 1].strip()
                    page_en_articles[art_num] = body

            # Split Arabic articles
            ar_splits = AR_ART_RE.split(right_text)
            page_ar_articles = {}
            if len(ar_splits) > 1:
                for i in range(1, len(ar_splits), 2):
                    raw_num = ar_splits[i]
                    norm_num = normalize_arabic_digits(raw_num)
                    body = ar_splits[i + 1].strip()
                    if norm_num.isdigit():
                        val = int(norm_num)
                        val_rev = int(norm_num[::-1])
                        # If normal value or reversed value matches an English article on this page
                        if val in page_en_articles:
                            page_ar_articles[val] = body
                        elif val_rev in page_en_articles:
                            page_ar_articles[val_rev] = body
                        elif 1 <= val <= 1149:
                            page_ar_articles[val] = body
                        elif 1 <= val_rev <= 1149:
                            page_ar_articles[val_rev] = body

            # Combine English and Arabic for each article found on this page
            for art_num, en_text in page_en_articles.items():
                if art_num in articles and articles[art_num].is_repealed:
                    # Already recorded as repealed; retain repeal flag
                    continue

                ar_text = page_ar_articles.get(art_num, "")
                # If Arabic text wasn't found by exact number key, check sequential position
                if not ar_text and page_ar_articles:
                    # Try matching by closest available key
                    if art_num in page_ar_articles:
                        ar_text = page_ar_articles[art_num]

                articles[art_num] = LegalArticle(
                    document_id=self.document_id,
                    jurisdiction=self.jurisdiction,
                    law_type=self.law_type,
                    law_name_ar=self.law_name_ar,
                    law_name_en=self.law_name_en,
                    law_year=self.law_year,
                    article_number=art_num,
                    book=current_book,
                    chapter=current_chapter,
                    section=current_section,
                    topic=current_topic,
                    text_ar=ar_text if ar_text else f"المادة {art_num}",
                    text_en=en_text if en_text else f"Article {art_num}",
                    is_repealed=False,
                    repeal_source="",
                    source_page=page_num,
                    citation=f"{self.law_name_en}, Article {art_num}"
                )

        # Sort all articles by article_number
        sorted_articles = [articles[k] for k in sorted(articles.keys())]
        return sorted_articles

    def _extract_repeal_notices(
        self,
        page_text: str,
        page_num: int,
        book: str,
        chapter: str,
        section: str,
        articles_dict: dict[int, LegalArticle]
    ) -> None:
        """Identify repeal ranges from source document notices and create flagged records."""
        # Check English repeal notices: e.g. "Articles 54-80 have been repealed"
        en_matches = EN_REPEAL_RE.findall(page_text)
        for s_str, e_str in en_matches:
            s_num = int(s_str)
            e_num = int(e_str)
            for n in range(s_num, e_num + 1):
                if n not in articles_dict or not articles_dict[n].is_repealed:
                    articles_dict[n] = LegalArticle(
                        document_id=self.document_id,
                        jurisdiction=self.jurisdiction,
                        law_type=self.law_type,
                        law_name_ar=self.law_name_ar,
                        law_name_en=self.law_name_en,
                        law_year=self.law_year,
                        article_number=n,
                        book=book,
                        chapter=chapter,
                        section=section,
                        topic="",
                        text_ar=f"ألغيت المواد من {s_num} إلى {e_num} بموجب تشريع لاحق.",
                        text_en=f"Articles {s_num}-{e_num} have been repealed by subsequent decree.",
                        is_repealed=True,
                        repeal_source=f"Source notice on page {page_num}: Articles {s_num}-{e_num} repealed",
                        source_page=page_num,
                        citation=f"{self.law_name_en}, Article {n} (Repealed)"
                    )

        # Check Arabic repeal notices: e.g. "ألغيت المواد من 54 إلى 80"
        ar_matches = AR_REPEAL_RE.findall(page_text)
        for s_str, e_str in ar_matches:
            s_norm = normalize_arabic_digits(s_str)
            e_norm = normalize_arabic_digits(e_str)
            if s_norm.isdigit() and e_norm.isdigit():
                s_num = int(s_norm)
                e_num = int(e_norm)
                # Handle possible reversed digits in Arabic notice
                if s_num > e_num:
                    s_num_rev = int(s_norm[::-1])
                    e_num_rev = int(e_norm[::-1])
                    if s_num_rev <= e_num_rev:
                        s_num, e_num = s_num_rev, e_num_rev
                    else:
                        s_num, e_num = min(s_num, e_num), max(s_num, e_num)

                for n in range(s_num, e_num + 1):
                    if n not in articles_dict or not articles_dict[n].is_repealed:
                        articles_dict[n] = LegalArticle(
                            document_id=self.document_id,
                            jurisdiction=self.jurisdiction,
                            law_type=self.law_type,
                            law_name_ar=self.law_name_ar,
                            law_name_en=self.law_name_en,
                            law_year=self.law_year,
                            article_number=n,
                            book=book,
                            chapter=chapter,
                            section=section,
                            topic="",
                            text_ar=f"ألغيت المواد من {s_num} إلى {e_num} بموجب تشريع لاحق.",
                            text_en=f"Articles {s_num}-{e_num} have been repealed by subsequent decree.",
                            is_repealed=True,
                            repeal_source=f"Source notice on page {page_num}: ألغيت المواد من {s_num} إلى {e_num}",
                            source_page=page_num,
                            citation=f"{self.law_name_en}, Article {n} (Repealed)"
                        )
