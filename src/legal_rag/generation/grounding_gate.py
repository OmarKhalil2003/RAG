from pydantic import BaseModel, Field
from legal_rag.models import RetrievalResult, QuerySignals


class GroundingDecision(BaseModel):
    passed: bool
    final_candidates: list[RetrievalResult] = Field(default_factory=list)
    refusal_reason: str | None = None


class LegalGroundingGate:
    """
    Deterministic hard gate enforced before LLM generation.
    Validates candidate relevance, legal scope, and requested article presence.
    """

    def __init__(self, min_relevance_floor: float = 0.02):
        self.min_relevance_floor = min_relevance_floor

    def evaluate(
        self,
        query_signals: QuerySignals,
        candidates: list[RetrievalResult],
        indexed_article_range: list[int] | None = None
    ) -> GroundingDecision:
        law_type = (query_signals.law_type or "civil").lower()
        if law_type == "arbitration":
            default_range = [1, 58]
            statute_name_ar = "قانون التحكيم المصري (رقم 27 لسنة 1994)"
            statute_name_en = "Egyptian Arbitration Law (Law No. 27 of 1994)"
        else:
            default_range = [1, 1149]
            statute_name_ar = "القانون المدني المصري"
            statute_name_en = "Egyptian Civil Code"

        indexed_range = indexed_article_range or default_range

        # 0. Check legal branch domain mismatch (e.g. criminal/penal questions when scope is civil/arbitration law)
        CRIMINAL_TERMS = ["عقوبة", "القتل", "سرقة", "حبس", "سجن", "إعدام", "جناية", "جنحة", "murder", "theft", "criminal penalty", "penal"]
        if law_type in ("civil", "arbitration"):
            q_norm = query_signals.normalized_query.lower()
            if any(term in q_norm for term in CRIMINAL_TERMS):
                if query_signals.language == "ar":
                    reason = f"قاعدة المعرفة الحالية لا تحتوي على معلومات كافية للإجابة عن هذا السؤال، حيث يتعلق بقانون العقوبات والمسائل الجنائية وهو خارج نطاق {statute_name_ar}."
                else:
                    reason = f"The current knowledge base does not contain sufficient information to answer this question, as it concerns criminal or penal law which is outside the scope of {statute_name_en}."
                return GroundingDecision(passed=False, final_candidates=[], refusal_reason=reason)

        # 1. Check if an explicit article number was queried
        if query_signals.article_numbers:
            for requested_num in query_signals.article_numbers:
                if requested_num < indexed_range[0] or requested_num > indexed_range[1]:
                    if query_signals.language == "ar":
                        reason = f"المادة المطلوبة ({requested_num}) غير موجودة ضمن نصوص {statute_name_ar} المتاح (نطاق المواد {indexed_range[0]} إلى {indexed_range[1]})."
                    else:
                        reason = f"The requested article ({requested_num}) does not exist in the available {statute_name_en} (articles range from {indexed_range[0]} to {indexed_range[1]})."
                    return GroundingDecision(passed=False, final_candidates=[], refusal_reason=reason)

            # Ensure requested article is present in candidates
            matching_candidates = [c for c in candidates if c.article_number in query_signals.article_numbers]
            if not matching_candidates:
                if query_signals.language == "ar":
                    reason = f"لم يتم العثور على أحكام المادة {query_signals.article_numbers} في قاعدة المعرفة المتاحة."
                else:
                    reason = f"Provisions for Article {query_signals.article_numbers} could not be retrieved from the available knowledge base."
                return GroundingDecision(passed=False, final_candidates=[], refusal_reason=reason)

        # 2. Check candidate pool existence
        if not candidates:
            if query_signals.language == "ar":
                reason = f"قاعدة المعرفة الحالية لا تحتوي على معلومات كافية للإجابة عن هذا السؤال، حيث تقتصر على {statute_name_ar}."
            else:
                reason = f"The current knowledge base does not contain sufficient information to answer this question from available legal sources ({statute_name_en})."
            return GroundingDecision(passed=False, final_candidates=[], refusal_reason=reason)

        # 3. Scope validation (jurisdiction & law_type)
        valid_candidates = []
        for c in candidates:
            meta = c.metadata or {}
            c_jurisdiction = meta.get("jurisdiction", "Egypt")
            c_law_type = meta.get("law_type", "civil")

            if query_signals.jurisdiction and c_jurisdiction.lower() != query_signals.jurisdiction.lower():
                continue
            if query_signals.law_type and c_law_type.lower() != query_signals.law_type.lower():
                continue

            valid_candidates.append(c)

        if not valid_candidates:
            if query_signals.language == "ar":
                reason = "لا توجد نصوص قانونية مطابقة لنطاق الاختصاص أو فرع القانون المطلوب في قاعدة المعرفة الحالية."
            else:
                reason = "No legal provisions matching the specified jurisdiction or law type were found in the current knowledge base."
            return GroundingDecision(passed=False, final_candidates=[], refusal_reason=reason)

        # 4. Relevance floor check for purely semantic queries
        is_exact = any("exact" in c.sources for c in valid_candidates)
        if not is_exact and valid_candidates[0].score < self.min_relevance_floor:
            if query_signals.language == "ar":
                reason = "النصوص المسترجعة لا ترقى لدرجة الصلة القانونية الكافية بهذا السؤال، ولا يجوز استنتاج أحكام قانونية غير مؤكدة."
            else:
                reason = "Retrieved legal provisions do not meet the minimum relevance threshold for this query. Unsupported legal conclusions are avoided."
            return GroundingDecision(passed=False, final_candidates=[], refusal_reason=reason)

        # All checks passed: provide exclusive candidates
        return GroundingDecision(
            passed=True,
            final_candidates=valid_candidates,
            refusal_reason=None
        )
