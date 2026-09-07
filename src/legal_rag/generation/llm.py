import re
from typing import Protocol, Iterator
from legal_rag.config import settings


class LLMClient(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate response from system and user prompt."""
        ...

    def generate_stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Stream response tokens from system and user prompt."""
        ...


class OpenAILLMClient:
    """OpenAI API Client Adapter."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self.client = None

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.client:
            raise ValueError("OpenAI client not configured or OPENAI_API_KEY missing.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4096
        )
        return response.choices[0].message.content or ""

    def generate_stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        if not self.client:
            raise ValueError("OpenAI client not configured or OPENAI_API_KEY missing.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4096,
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                yield delta


class MockLLMClient:
    """
    Deterministic mock generator used to test retrieval, citation, cache,
    and pipeline behavior without an external LLM dependency.
    """

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        # Detect language from user prompt
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in user_prompt)
        
        # Check if context contains repealed articles
        has_repealed = "REPEALED" in user_prompt or "ملغاة" in user_prompt
        
        # Extract cited articles from context block in user prompt
        art_matches = re.findall(r'Article\s+(\d+)', user_prompt, re.IGNORECASE)
        found_articles = sorted(list(set(int(m) for m in art_matches)))
        
        # Detect role persona
        role_label = "Lawyer"
        if "Regular Citizen" in system_prompt or "مواطن" in system_prompt:
            role_label = "Citizen"
        elif "Law Student" in system_prompt or "طالب" in system_prompt:
            role_label = "Law Student"
        elif "Legal Researcher" in system_prompt or "باحث" in system_prompt:
            role_label = "Legal Researcher"

        # Check for specific known articles to provide realistic legal answers
        if has_repealed:
            rep_arts = ", ".join(str(a) for a in found_articles)
            if is_arabic:
                return (
                    f"بناءً على نصوص القانون المدني المصري المسترجعة، نود إحاطتكم بأن المادة ({rep_arts}) "
                    f"قد تم إلغاؤها بموجب تشريع لاحق ولا تعتبر مادة قانونية سارية المفعول في الوقت الحالي. "
                    f"(السند القانوني: القانون المدني المصري، مادة {rep_arts})."
                )
            else:
                return (
                    f"Based on the retrieved provisions of the Egyptian Civil Code, please note that Article ({rep_arts}) "
                    f"has been repealed by subsequent decree and is no longer an active statutory provision. "
                    f"(Legal Basis: Egyptian Civil Code, Article {rep_arts})."
                )

        if 147 in found_articles:
            if is_arabic:
                if role_label == "Lawyer":
                    return (
                        "وفقاً للمادة 147 من القانون المدني المصري: العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين "
                        "أو للأسباب التي يقررها القانون. ومع ذلك، إذا طرأت حوادث استثنائية عامة لم يكن في الوسع توقعها وترتب على حدوثها "
                        "أن تنفيذ الالتزام التعاقدي صار مرهقاً للمدين، جاز للقاضي تبعا للظروف وسد الموازنة بين مصلحة الطرفين أن يرد الالتزام "
                        "المرهق إلى الحد المعقول. (السند القانوني: القانون المدني المصري، المادة 147)."
                    )
                elif role_label == "Citizen":
                    return (
                        "ببساطة، الاتفاق بين الطرفين له قوة القانون بينهما (العقد شريعة المتعاقدين). لا يحق لأي طرف تغييره إلا بموافقة الطرف الآخر. "
                        "ولكن إذا حدثت ظروف طارئة عامة غير متوقعة جعلت التنفيذ مرهقاً وخسارته فادحة، يمكن للقاضي تخفيف هذا الالتزام لحمايتك. "
                        "(المرجع: القانون المدني المصري، المادة 147)."
                    )
                elif role_label == "Law Student":
                    return (
                        "تكرس المادة 147 من القانون المدني مبدأ القوة الملزمة للعقد (العقد شريعة المتعاقدين)، والاستثناء الهام الوارد عليه "
                        "وهو (نظرية الظروف الطارئة) بشروطها: حدوث ظرف استثنائي عام، غير متوقع، يجعل التنفيذ مرهقاً لا مستحيلاً. "
                        "(المستند: القانون المدني المصري، المادة 147)."
                    )
                else:  # Legal Researcher
                    return (
                        "بالرجوع إلى القسم الأول (الالتزامات)، الباب الأول (مصادر الالتزام)، الفصل الأول (العقد): تنص المادة 147 على "
                        "مبدأ القوة الملزمة وآثار العقد بالنسبة للمتعاقدين، مقترنة بضابط نظرية الظروف الطارئة المقيدة لحرية التعاقد حماية لمبدأ العدالة. "
                        "(السند التشريعي: القانون المدني المصري، المادة 147)."
                    )
            else:
                return (
                    "Under Article 147 of the Egyptian Civil Code, the contract makes the law of the parties. It cannot be revoked or altered "
                    "except by mutual consent or for reasons provided by law. In cases of exceptional and unpredictable events of a general character "
                    "that render performance excessively onerous, the court may reduce obligations to reasonable limits. "
                    "(Citation: Egyptian Civil Code, Article 147)."
                )

        if 157 in found_articles:
            if is_arabic:
                return (
                    "طبقاً لنص المادة 157 من القانون المدني المصري: في العقود الملزمة للجانبين، إذا لم يوف أحد المتعاقدين بالتزامه "
                    "جاز للمتعاقد الآخر بعد إعذار المدين أن يطالب بتنفيذ العقد أو بفسخه، مع التعويض إن كان له مقتض. "
                    "(السند القانوني: القانون المدني المصري، المادة 157)."
                )
            else:
                return (
                    "Pursuant to Article 157 of the Egyptian Civil Code: In bilateral contracts, if one of the parties fails to perform his obligation, "
                    "the other party may, after serving a formal summons, demand either the performance of the contract or its rescission, with damages if due. "
                    "(Citation: Egyptian Civil Code, Article 157)."
                )

        # General deterministic fallback from found articles
        cited_str = ", ".join(str(a) for a in found_articles)
        if is_arabic:
            return (
                f"وفقاً للنصوص القانونية المسترجعة من القانون المدني المصري (المادة {cited_str})، "
                f"فإن الأحكام المنظمة لهذا الموضوع محددة بنص القانون كما وردت في المواد المسترجعة. "
                f"(السند القانوني: القانون المدني المصري، المواد {cited_str})."
            )
        else:
            return (
                f"According to the retrieved provisions of the Egyptian Civil Code (Article {cited_str}), "
                f"the applicable legal rules are set forth in the retrieved articles. "
                f"(Legal basis: Egyptian Civil Code, Articles {cited_str})."
            )

    def generate_stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        full_text = self.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


_UNSET = object()


class GeminiLLMClient:
    """Google Gemini LLM Adapter."""

    def __init__(self, api_key: str | None = _UNSET, model_name: str | None = None):
        import os
        if api_key is _UNSET:
            self.api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        else:
            self.api_key = api_key
        self.model_name = model_name or settings.gemini_model or "gemini-3.1-flash-lite"
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file "
                "or enter it directly in the Streamlit sidebar."
            )
        import google.generativeai as genai
        
        # Build candidate fallback sequence starting with preferred model
        preferred = (self.model_name or "gemini-2.0-flash").strip()
        candidates = [preferred]
        for c in [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-flash-latest",
            "gemini-2.5-pro",
            "gemini-1.5-pro"
        ]:
            if c not in candidates:
                candidates.append(c)

        recoverable_keywords = [
            "not found", "no longer available", "unsupported", "404",
            "resourceexhausted", "quota", "rate", "overloaded",
            "serviceunavailable", "unavailable", "503", "500", "502", "504",
            "internal", "deadline", "timeout", "connection", "reset", "grpc"
        ]

        last_error = None
        for candidate in candidates:
            try:
                clean_name = candidate.replace("models/", "")
                model = genai.GenerativeModel(
                    model_name=clean_name,
                    system_instruction=system_prompt
                )
                response = model.generate_content(
                    user_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=4096
                    )
                )
                if response and response.text:
                    self.model_name = clean_name
                    return response.text
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                if any(x in err_msg for x in recoverable_keywords):
                    import time
                    time.sleep(0.3)
                    continue
                raise e

        if last_error:
            return f"⚠️ Service Notice: Google AI temporarily unavailable ({last_error}). Please retry."
        return ""

    def generate_stream(self, *, system_prompt: str, user_prompt: str) -> Iterator[str]:
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file "
                "or Streamlit Secrets."
            )
        import google.generativeai as genai

        preferred = (self.model_name or "gemini-2.0-flash").strip()
        candidates = [preferred]
        for c in [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-flash-latest",
            "gemini-2.5-pro",
            "gemini-1.5-pro"
        ]:
            if c not in candidates:
                candidates.append(c)

        recoverable_keywords = [
            "not found", "no longer available", "unsupported", "404",
            "resourceexhausted", "quota", "rate", "overloaded",
            "serviceunavailable", "unavailable", "503", "500", "502", "504",
            "internal", "deadline", "timeout", "connection", "reset", "grpc"
        ]

        last_error = None
        for candidate in candidates:
            yielded_any = False
            try:
                clean_name = candidate.replace("models/", "")
                model = genai.GenerativeModel(
                    model_name=clean_name,
                    system_instruction=system_prompt
                )
                response = model.generate_content(
                    user_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=4096
                    ),
                    stream=True
                )
                for chunk in response:
                    if chunk and chunk.text:
                        yielded_any = True
                        yield chunk.text
                if yielded_any:
                    self.model_name = clean_name
                    return
            except Exception as e:
                last_error = e
                # If we already yielded tokens before failure, inform user rather than silent cutoff
                if yielded_any:
                    yield f"\n\n*(تنبيه: انقطع بث الإجابة بسبب انقطاع في الاتصال: {e})*"
                    return
                err_msg = str(e).lower()
                if any(x in err_msg for x in recoverable_keywords):
                    import time
                    time.sleep(0.3)
                    continue
                raise e

        if last_error:
            yield f"⚠️ عذراً، تعذر إكمال الإجابة من Google AI نظراً لضغط مؤقت على الخدمة ({last_error}). يرجى إعادة المحاولة."


def get_llm_client(provider: str | None = None, api_key: str | None = None, model: str | None = None) -> LLMClient:
    """Factory function to get the configured LLM client."""
    import os
    selected_provider = (provider or settings.llm_provider).lower()
    
    if selected_provider == "gemini":
        key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if key:
            return GeminiLLMClient(api_key=key, model_name=model)
        # If key is absent, return GeminiLLMClient if explicitly chosen so it prompts for key,
        # or return MockLLMClient if local dev mode
        if settings.local_dev_mode:
            return MockLLMClient()
        return GeminiLLMClient(api_key=None, model_name=model)
    elif selected_provider == "openai":
        key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if key:
            return OpenAILLMClient(api_key=key, model=model)
        return MockLLMClient()
    
    return MockLLMClient()
