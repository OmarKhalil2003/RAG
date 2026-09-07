import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_rag.retrieval.query_parser import QueryParser
from legal_rag.retrieval.embeddings import BGEM3Embedder
from legal_rag.retrieval.qdrant import QdrantLegalStore
from legal_rag.retrieval.sparse import BM25SparseRetriever
from legal_rag.retrieval.hybrid import reciprocal_rank_fusion
from legal_rag.retrieval.reranker import LegalReranker


def run_ablation():
    print("=" * 70)
    print("RUNNING RETRIEVAL ABLATION BENCHMARK")
    print("=" * 70)
    print("Comparing configurations:")
    print("  1. Dense Only (BGE-M3)")
    print("  2. Dense + Sparse (BM25)")
    print("  3. Dense + Sparse + Exact Lookup (RRF + Priority Injection)")
    print("  4. Full Pipeline: Dense + Sparse + Exact + BGE Reranker")
    print("-" * 70)

    dataset_path = project_root / "evaluation" / "questions.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = [c for c in json.load(f) if c.get("type") != "out_of_scope" and c.get("expected_articles")]

    parser = QueryParser()
    embedder = BGEM3Embedder()
    qdrant_store = QdrantLegalStore()
    bm25 = BM25SparseRetriever.load(project_root / "data" / "processed" / "bm25_index.pkl")
    reranker = LegalReranker()

    configs = [
        "1. Dense Only",
        "2. Dense + Sparse",
        "3. Dense + Sparse + Exact",
        "4. Dense + Sparse + Exact + Reranker"
    ]

    recall_at_1 = {c: 0 for c in configs}
    recall_at_5 = {c: 0 for c in configs}
    exact_accuracy = {c: 0 for c in configs}
    exact_cases = sum(1 for c in cases if c.get("type") in ("exact_article", "arabic_indic_numeral"))

    for item in cases:
        q = item["question"]
        target = set(item["expected_articles"])
        is_exact = item.get("type") in ("exact_article", "arabic_indic_numeral")

        signals = parser.parse(q, jurisdiction="Egypt", law_type="civil")
        q_vec = embedder.encode_query(signals.normalized_query)

        # Config 1: Dense Only
        res_dense = qdrant_store.dense_search(q_vec, limit=20)
        c1_articles = [r.article_number for r in res_dense]

        # Config 2: Dense + Sparse
        res_sparse = bm25.search(signals.normalized_query, limit=20)
        res_c2 = reciprocal_rank_fusion(res_dense, res_sparse, exact_results=None, limit=20)
        c2_articles = [r.article_number for r in res_c2]

        # Config 3: Dense + Sparse + Exact Lookup
        exact_results = []
        if signals.article_numbers:
            for num in signals.article_numbers:
                exact_results.extend(qdrant_store.exact_lookup(num))
        res_c3 = reciprocal_rank_fusion(res_dense, res_sparse, exact_results=exact_results, limit=20)
        c3_articles = [r.article_number for r in res_c3]

        # Config 4: Dense + Sparse + Exact + Reranker
        res_c4 = reranker.rerank(q, res_c3, top_k=5)
        c4_articles = [r.article_number for r in res_c4]

        # Evaluate
        for name, arts in [
            ("1. Dense Only", c1_articles),
            ("2. Dense + Sparse", c2_articles),
            ("3. Dense + Sparse + Exact", c3_articles),
            ("4. Dense + Sparse + Exact + Reranker", c4_articles)
        ]:
            if arts and arts[0] in target:
                recall_at_1[name] += 1
            if any(a in target for a in arts[:5]):
                recall_at_5[name] += 1
            if is_exact and arts and arts[0] in target:
                exact_accuracy[name] += 1

    n = len(cases)
    print("\n" + "=" * 70)
    print(f"{'Configuration':<38} | {'Recall@1':<10} | {'Recall@5':<10} | {'Exact Acc':<10}")
    print("-" * 70)
    for c in configs:
        r1_pct = (recall_at_1[c] / n) * 100
        r5_pct = (recall_at_5[c] / n) * 100
        ex_pct = (exact_accuracy[c] / exact_cases) * 100 if exact_cases else 0.0
        print(f"{c:<38} | {r1_pct:>8.1f}% | {r5_pct:>8.1f}% | {ex_pct:>8.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation()
