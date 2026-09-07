import re
import time
from pathlib import Path
from typing import Callable, Iterator

from legal_rag.config import settings
from legal_rag.models import (
    UserRole,
    QuerySignals,
    SourceCitation,
    RAGResponse,
    RetrievalResult
)
from legal_rag.retrieval.query_parser import QueryParser
from legal_rag.retrieval.embeddings import BGEM3Embedder
from legal_rag.retrieval.qdrant import QdrantLegalStore
from legal_rag.retrieval.sparse import BM25SparseRetriever
from legal_rag.retrieval.hybrid import reciprocal_rank_fusion
from legal_rag.retrieval.reranker import LegalReranker
from legal_rag.generation.grounding_gate import LegalGroundingGate
from legal_rag.generation.prompts import construct_prompt
from legal_rag.generation.llm import get_llm_client, LLMClient
from legal_rag.cache.semantic_cache import RedisSemanticCache
from legal_rag.ingestion.normalizer import clean_arabic_ocr_artifacts


class RAGService:
    """
    Core Legal RAG application service orchestrating:
    Query Parsing -> Semantic Cache -> Hybrid Retrieval -> RRF -> Reranking
    -> Grounding Gate -> Grounded Role Generation -> Deterministic Citations.
    Supports streaming generation and conversational context.
    """

    def __init__(
        self,
        qdrant_store: QdrantLegalStore | None = None,
        sparse_retriever: BM25SparseRetriever | None = None,
        reranker: LegalReranker | None = None,
        embedder: BGEM3Embedder | None = None,
        cache: RedisSemanticCache | None = None,
        llm_client: LLMClient | None = None
    ):
        self.query_parser = QueryParser()
        self.qdrant_store = qdrant_store or QdrantLegalStore()
        
        # Load BM25 index
        bm25_path = Path("data/processed/bm25_index.pkl")
        if sparse_retriever:
            self.sparse_retriever = sparse_retriever
        elif bm25_path.exists():
            self.sparse_retriever = BM25SparseRetriever.load(bm25_path)
        else:
            self.sparse_retriever = BM25SparseRetriever()

        self.reranker = reranker or LegalReranker()
        self.embedder = embedder or BGEM3Embedder()
        self.cache = cache or RedisSemanticCache()
        self.llm_client = llm_client or get_llm_client()
        self.grounding_gate = LegalGroundingGate()

    def _enrich_query_with_history(self, question: str, history: list[dict] | None) -> str:
        """Enriches referential follow-up questions with context from prior turns."""
        if not history:
            return question

        # If current question already contains explicit article numbers, no enrichment needed
        temp_signals = self.query_parser.parse(question)
        if temp_signals.article_numbers:
            return question

        # Check last user and assistant messages
        last_user = ""
        last_asst = ""
        for msg in reversed(history):
            if msg.get("role") == "user" and not last_user:
                last_user = msg.get("content", "")
            elif msg.get("role") == "assistant" and not last_asst:
                last_asst = msg.get("content", "")
            if last_user and last_asst:
                break

        if not last_user:
            return question

        # Extract articles or topic from last query
        prev_signals = self.query_parser.parse(last_user)
        if prev_signals.article_numbers:
            art_str = f"المادة {prev_signals.article_numbers[0]}" if prev_signals.language == "ar" else f"Article {prev_signals.article_numbers[0]}"
            return f"{art_str} {question}"

        # Otherwise prepend first 40 chars of previous subject
        clean_prev = re.sub(r'[\?\.\!؟]', '', last_user).strip()
        if len(clean_prev) > 40:
            clean_prev = clean_prev[:40]
        return f"{clean_prev} {question}"

    def ask_stream(
        self,
        question: str,
        role: UserRole = UserRole.LAWYER,
        jurisdiction: str = "Egypt",
        law_type: str = "civil",
        history: list[dict] | None = None
    ) -> tuple[RAGResponse, Iterator[str], Callable[[str], RAGResponse]]:
        """
        Executes hybrid retrieval, reranking, and returns:
        1. Prepared RAGResponse metadata (sources, citations, query_signals, cached).
        2. Generator yielding streaming token chunks.
        3. Finalize callback taking the full assembled text to update response, record latency, and cache.
        """
        start_time = time.perf_counter()

        # Contextually enrich follow-up questions
        enriched_query = self._enrich_query_with_history(question, history)

        # 1. Parse query
        signals = self.query_parser.parse(
            query=enriched_query,
            jurisdiction=jurisdiction,
            law_type=law_type
        )

        # 2. Encode query vector
        query_vector = self.embedder.encode_query(signals.normalized_query)

        # 3. Check Semantic Cache
        cached_resp = self.cache.get(
            query_vector=query_vector,
            jurisdiction=jurisdiction,
            law_type=law_type,
            user_role=role,
            normalized_query=signals.normalized_query,
            article_numbers=signals.article_numbers
        )
        if cached_resp:
            cached_resp.latency_ms = (time.perf_counter() - start_time) * 1000.0
            def finalize_cached(text: str) -> RAGResponse:
                return cached_resp
            def stream_cached() -> Iterator[str]:
                yield cached_resp.answer
            return cached_resp, stream_cached(), finalize_cached

        # 4. Exact article lookup
        exact_candidates: list[RetrievalResult] = []
        if signals.article_numbers:
            for art_num in signals.article_numbers:
                exact_candidates.extend(
                    self.qdrant_store.exact_lookup(
                        article_number=art_num,
                        jurisdiction=jurisdiction,
                        law_type=law_type
                    )
                )

        # 5. Dense semantic retrieval
        dense_candidates = self.qdrant_store.dense_search(
            query_vector=query_vector,
            limit=20,
            jurisdiction=jurisdiction,
            law_type=law_type
        )

        # 6. Sparse lexical retrieval
        sparse_candidates = self.sparse_retriever.search(
            query=signals.normalized_query,
            limit=20,
            jurisdiction=jurisdiction,
            law_type=law_type
        )

        # 7. Reciprocal Rank Fusion + Deterministic Exact Priority Injection
        candidate_pool = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            exact_results=exact_candidates,
            k=60,
            limit=20
        )

        # 8. Cross-encoder reranking
        top_candidates = self.reranker.rerank(
            query=question,
            candidates=candidate_pool,
            top_k=5
        )

        # 9. Grounding Gate validation
        decision = self.grounding_gate.evaluate(
            query_signals=signals,
            candidates=top_candidates
        )

        if not decision.passed:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            refusal_resp = RAGResponse(
                answer=decision.refusal_reason or "Insufficient legal information available in corpus.",
                sources=[],
                cached=False,
                retrieval_count=0,
                query_signals=signals,
                latency_ms=latency_ms
            )
            def finalize_refusal(text: str) -> RAGResponse:
                return refusal_resp
            def stream_refusal() -> Iterator[str]:
                yield refusal_resp.answer
            return refusal_resp, stream_refusal(), finalize_refusal

        final_candidates = decision.final_candidates

        # 10. Role-aware prompt construction (with conversation history)
        try:
            sys_prompt, usr_prompt = construct_prompt(
                question=question,
                candidates=final_candidates,
                role=role,
                language=signals.language,
                history=history
            )
        except TypeError as te:
            if "history" in str(te):
                import importlib
                import legal_rag.generation.prompts as _prompts_mod
                importlib.reload(_prompts_mod)
                sys_prompt, usr_prompt = _prompts_mod.construct_prompt(
                    question=question,
                    candidates=final_candidates,
                    role=role,
                    language=signals.language,
                    history=history
                )
            else:
                raise

        # Build citations
        citations: list[SourceCitation] = []
        for c in final_candidates:
            meta = c.metadata or {}
            raw_hier = " > ".join(filter(None, [meta.get("book"), meta.get("chapter"), meta.get("section")]))
            clean_hier = clean_arabic_ocr_artifacts(raw_hier) if raw_hier else ""
            clean_text = clean_arabic_ocr_artifacts(c.text_ar) if c.text_ar else ""
            citations.append(
                SourceCitation(
                    document_id=meta.get("document_id", "egyptian_civil_code_1948"),
                    law_name=meta.get("law_name_ar" if signals.language == "ar" else "law_name_en", "Egyptian Civil Code"),
                    article_number=c.article_number,
                    citation=c.citation,
                    is_repealed=c.is_repealed,
                    text_ar=clean_text,
                    text_en=c.text_en,
                    hierarchy=clean_hier
                )
            )

        retrieval_latency_ms = (time.perf_counter() - start_time) * 1000.0
        prepared_resp = RAGResponse(
            answer="",
            sources=citations,
            cached=False,
            retrieval_count=len(citations),
            query_signals=signals,
            latency_ms=retrieval_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms
        )

        def stream_tokens() -> Iterator[str]:
            if hasattr(self.llm_client, "generate_stream"):
                for chunk in self.llm_client.generate_stream(
                    system_prompt=sys_prompt,
                    user_prompt=usr_prompt
                ):
                    yield chunk
            else:
                yield self.llm_client.generate(
                    system_prompt=sys_prompt,
                    user_prompt=usr_prompt
                )

        def finalize_response(full_answer: str) -> RAGResponse:
            total_latency = (time.perf_counter() - start_time) * 1000.0
            prepared_resp.answer = full_answer
            prepared_resp.latency_ms = total_latency
            prepared_resp.retrieval_latency_ms = retrieval_latency_ms

            # Cache completed response
            self.cache.store(
                query=question,
                query_vector=query_vector,
                response=prepared_resp,
                jurisdiction=jurisdiction,
                law_type=law_type,
                user_role=role,
                normalized_query=signals.normalized_query,
                article_numbers=signals.article_numbers
            )
            return prepared_resp

        return prepared_resp, stream_tokens(), finalize_response

    def ask(
        self,
        question: str,
        role: UserRole = UserRole.LAWYER,
        jurisdiction: str = "Egypt",
        law_type: str = "civil",
        history: list[dict] | None = None
    ) -> RAGResponse:
        prepared_resp, token_stream, finalize = self.ask_stream(
            question=question,
            role=role,
            jurisdiction=jurisdiction,
            law_type=law_type,
            history=history
        )
        full_text = "".join(token_stream)
        return finalize(full_text)

