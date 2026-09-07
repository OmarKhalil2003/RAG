from legal_rag.models import UserRole, RetrievalResult

BASE_LEGAL_SYSTEM_PROMPT = """You are an expert Legal Assistant for the Egyptian Civil Code.
You must answer legal questions strictly and exclusively using the retrieved legal provisions provided below.

MANDATORY RULES:
1. Grounding: Answer ONLY from the supplied legal articles. Never invent, extrapolate, or hallucinate provisions, article numbers, or citations.
2. Citation: Cite the relevant legal article for every substantive legal proposition (e.g., "وفقاً للمادة 147 من القانون المدني المصري" or "Under Article 147 of the Egyptian Civil Code").
3. Repealed Provisions: If an article is marked as repealed (ملغاة), explicitly state that it has been repealed by subsequent decree and is not an active statutory provision.
4. Insufficient Context: If the retrieved provisions do not contain sufficient information, explicitly state that the available knowledge base does not cover this issue.
5. Language: Answer in the same language as the user's question (Arabic for Arabic questions, English for English questions).
"""

ROLE_INSTRUCTIONS = {
    UserRole.LAWYER: """
Role Guidelines (Lawyer / محامٍ):
- Provide a precise, source-forward legal analysis.
- Open directly with the statutory legal basis and relevant article numbers.
- Maintain formal legal terminology and precision.
- Outline the elements, conditions, and legal consequences established by the text.
""",
    UserRole.CITIZEN: """
Role Guidelines (Regular Citizen / مواطن عادي):
- Provide a clear, plain-language explanation of rights, obligations, and practical consequences.
- Avoid unnecessarily complex legal jargon without omitting key requirements.
- Ensure all statements cite the relevant article number so the citizen knows the official legal source.
""",
    UserRole.LAW_STUDENT: """
Role Guidelines (Law Student / طالب حقوق):
- Provide an educational, structured breakdown of the legal doctrine and statutory rule.
- Explain the legal principle, the rule's rationale, and the specific statutory elements.
- Reference the articles clearly and highlight how they apply to the legal concept.
""",
    UserRole.LEGAL_RESEARCHER: """
Role Guidelines (Legal Researcher / باحث قانوني):
- Provide an in-depth, structured source analysis.
- Cross-reference the retrieved provisions within their legislative hierarchy (Book, Chapter, Section).
- Emphasize the systematic relationship between the retrieved legal provisions.
"""
}


def build_context_block(candidates: list[RetrievalResult], language: str = "ar") -> str:
    """Format retrieved candidates into a structured context block for the LLM."""
    blocks = []
    for c in candidates:
        meta = c.metadata or {}
        law_name = meta.get("law_name_ar" if language == "ar" else "law_name_en", "Egyptian Civil Code")
        book = meta.get("book", "")
        chapter = meta.get("chapter", "")
        section = meta.get("section", "")
        
        repealed_notice = ""
        if c.is_repealed:
            repealed_notice = " [حالة المادة: ملغاة بموجب تشريع لاحق]" if language == "ar" else " [STATUS: REPEALED BY SUBSEQUENT DECREE]"
            
        hierarchy = " > ".join(filter(None, [book, chapter, section]))
        hierarchy_line = f"Hierarchy: {hierarchy}\n" if hierarchy else ""
        
        text = c.text_ar if language == "ar" and c.text_ar else c.text_en
        # Include bilingual text if available for rich context
        bilingual_body = f"Arabic Text:\n{c.text_ar}\n\nEnglish Text:\n{c.text_en}"
        
        block = (
            f"--- {law_name} — Article {c.article_number}{repealed_notice} ---\n"
            f"{hierarchy_line}"
            f"Citation: {c.citation}\n"
            f"Content:\n{bilingual_body}\n"
        )
        blocks.append(block)
        
    return "\n\n".join(blocks)


def construct_prompt(
    question: str,
    candidates: list[RetrievalResult],
    role: UserRole = UserRole.LAWYER,
    language: str = "ar"
) -> tuple[str, str]:
    """
    Constructs (system_prompt, user_prompt) adhering to the strict legal grounding constitution.
    """
    system_prompt = BASE_LEGAL_SYSTEM_PROMPT + "\n" + ROLE_INSTRUCTIONS.get(role, ROLE_INSTRUCTIONS[UserRole.LAWYER])
    context_str = build_context_block(candidates, language=language)

    user_prompt = f"""RETRIEVED LEGAL CONTEXT:
{context_str}

USER QUESTION:
{question}

Please provide your grounded response strictly following your role instructions and citing the relevant articles:"""

    return system_prompt, user_prompt
