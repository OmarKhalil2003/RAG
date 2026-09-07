from pydantic import BaseModel, Field
from legal_rag.models import LegalArticle


class ValidationReport(BaseModel):
    expected_range: list[int]
    extracted_count: int
    validated_count: int
    missing_articles: list[int] = Field(default_factory=list)
    unexpected_articles: list[int] = Field(default_factory=list)
    repealed_articles: list[int] = Field(default_factory=list)
    is_valid: bool
    summary: str


class CorpusValidator:
    """
    Validates extracted legal articles against expected ranges and integrity rules.
    """

    def __init__(self, expected_range: list[int] | None = None):
        self.expected_range = expected_range or [1, 1149]

    def validate(self, articles: list[LegalArticle]) -> ValidationReport:
        expected_start, expected_end = self.expected_range[0], self.expected_range[1]
        expected_set = set(range(expected_start, expected_end + 1))
        
        extracted_nums = set(a.article_number for a in articles)
        missing = sorted(list(expected_set - extracted_nums))
        unexpected = sorted(list(extracted_nums - expected_set))
        
        repealed = sorted([a.article_number for a in articles if a.is_repealed])
        
        # Check basic content integrity
        valid_articles = 0
        for a in articles:
            if a.article_number > 0 and (a.text_ar or a.text_en or a.is_repealed):
                valid_articles += 1

        is_valid = len(missing) == 0 and len(articles) > 0

        summary = (
            f"Expected Range: {expected_start}–{expected_end} ({len(expected_set)} articles) | "
            f"Extracted: {len(articles)} | Validated: {valid_articles} | "
            f"Repealed: {len(repealed)} | Missing: {len(missing)} | Unexpected: {len(unexpected)}"
        )

        return ValidationReport(
            expected_range=self.expected_range,
            extracted_count=len(articles),
            validated_count=valid_articles,
            missing_articles=missing,
            unexpected_articles=unexpected,
            repealed_articles=repealed,
            is_valid=is_valid,
            summary=summary
        )
