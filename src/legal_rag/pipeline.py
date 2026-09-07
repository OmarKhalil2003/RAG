import time
from pathlib import Path

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

    def ask(
        self,
        question: str,
        role: UserRole = UserRole.LAWYER,
        jurisdiction: str = "Egypt",
        law_type: str = "civil"
    ) -> RAGResponse:
        start_time = time.perf_counter()

        # 1. Parse query
        signals = self.query_parser.parse(
            query=question,
            jurisdiction=jurisdiction,
            law_type=law_type
        )

        # 2. Encode query vector for semantic retrieval and caching
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
            return cached_resp

        # 4. Exact article lookup if explicit article numbers detected
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
            return RAGResponse(
                answer=decision.refusal_reason or "Insufficient legal information available in corpus.",
                sources=[],
                cached=False,
                retrieval_count=0,
                query_signals=signals,
                latency_ms=latency_ms
            )

        final_candidates = decision.final_candidates

        # 10. Role-aware prompt construction & LLM Generation
        sys_prompt, usr_prompt = construct_prompt(
            question=question,
            candidates=final_candidates,
            role=role,
            language=signals.language
        )

        answer = self.llm_client.generate(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt
        )

        # 11. Deterministic Citation Building directly from retrieved articles
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

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response = RAGResponse(
            answer=answer,
            sources=citations,
            cached=False,
            retrieval_count=len(citations),
            query_signals=signals,
            latency_ms=latency_ms
        )

        # 12. Store in semantic cache
        self.cache.store(
            query=question,
            query_vector=query_vector,
            response=response,
            jurisdiction=jurisdiction,
            law_type=law_type,
            user_role=role,
            normalized_query=signals.normalized_query,
            article_numbers=signals.article_numbers
        )

        return response
