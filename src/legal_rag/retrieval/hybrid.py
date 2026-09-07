from collections import defaultdict
from legal_rag.models import RetrievalResult


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    exact_results: list[RetrievalResult] | None = None,
    k: int = 60,
    limit: int = 20
) -> list[RetrievalResult]:
    """
    Fuse dense semantic and sparse lexical rankings using Reciprocal Rank Fusion (RRF),
    then inject deterministic exact article matches with top priority.
    
    Formula:
        RRF_Score(d) = sum(1 / (k + rank_m(d))) for m in {dense, sparse}
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    doc_map: dict[int, RetrievalResult] = {}
    sources_map: dict[int, set[str]] = defaultdict(set)

    # 1. Score dense candidates
    for rank, res in enumerate(dense_results, start=1):
        num = res.article_number
        rrf_scores[num] += 1.0 / (k + rank)
        doc_map[num] = res
        sources_map[num].add("dense")

    # 2. Score sparse candidates
    for rank, res in enumerate(sparse_results, start=1):
        num = res.article_number
        rrf_scores[num] += 1.0 / (k + rank)
        if num not in doc_map:
            doc_map[num] = res
        sources_map[num].add("sparse")

    # Sort candidates by RRF score
    sorted_articles = sorted(rrf_scores.keys(), key=lambda n: rrf_scores[n], reverse=True)
    fused_results: list[RetrievalResult] = []

    for num in sorted_articles:
        base_res = doc_map[num]
        fused_results.append(
            RetrievalResult(
                chunk_id=base_res.chunk_id,
                article_number=num,
                score=rrf_scores[num],
                sources=sorted(list(sources_map[num])),
                metadata=base_res.metadata,
                text_ar=base_res.text_ar,
                text_en=base_res.text_en,
                is_repealed=base_res.is_repealed,
                citation=base_res.citation
            )
        )

    # 3. Deterministic Exact Priority Injection:
    # If exact article matches were retrieved, ensure they appear at the very top of the candidate pool
    if exact_results:
        exact_nums = {r.article_number for r in exact_results}
        # Remove exact results from fused list if already present to avoid duplication
        remaining = [r for r in fused_results if r.article_number not in exact_nums]
        
        injected = []
        for exact_res in exact_results:
            existing_sources = sources_map.get(exact_res.article_number, set())
            existing_sources.add("exact")
            injected.append(
                RetrievalResult(
                    chunk_id=exact_res.chunk_id,
                    article_number=exact_res.article_number,
                    score=10.0,  # Explicit high priority score
                    sources=sorted(list(existing_sources)),
                    metadata=exact_res.metadata,
                    text_ar=exact_res.text_ar,
                    text_en=exact_res.text_en,
                    is_repealed=exact_res.is_repealed,
                    citation=exact_res.citation
                )
            )
        fused_results = injected + remaining

    return fused_results[:limit]
