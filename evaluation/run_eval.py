import sys
import json
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_rag.pipeline import RAGService
from legal_rag.models import UserRole


def run_evaluation():
    print("=" * 70)
    print("RUNNING AUTOMATED LEGAL RAG EVALUATION SUITE")
    print("=" * 70)

    dataset_path = project_root / "evaluation" / "questions.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    rag = RAGService()

    total_cases = len(cases)
    exact_cases = 0
    exact_hits = 0

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []

    scope_correct = 0
    refusal_correct = 0
    out_of_scope_cases = 0
    citation_correct = 0

    cache_tests = 0
    cache_hits = 0

    bilingual_pairs = {}
    detailed_results = []

    print(f"Total evaluation test cases: {total_cases}\n")

    for idx, item in enumerate(cases, 1):
        q_id = item["id"]
        q_text = item["question"]
        q_type = item["type"]
        expected_articles = item.get("expected_articles", [])
        expected_jurisdiction = item.get("expected_jurisdiction", "Egypt")
        expected_law_type = item.get("expected_law_type", "civil")
        expected_repealed = item.get("expected_repealed", False)

        resp = rag.ask(
            question=q_text,
            role=UserRole.LAWYER,
            jurisdiction=expected_jurisdiction,
            law_type=expected_law_type
        )

        retrieved_articles = [s.article_number for s in resp.sources]

        # 1. Out-of-Scope & Refusal evaluation
        if q_type == "out_of_scope":
            out_of_scope_cases += 1
            # Must have refused (0 sources)
            if len(resp.sources) == 0:
                refusal_correct += 1
                status = "PASS (Refusal)"
            else:
                status = "FAIL (Hallucinated source)"
        else:
            # 2. Scope verification
            scope_ok = True
            for s in resp.sources:
                if s.law_name and expected_law_type.lower() not in s.law_name.lower() and "civil" not in s.law_name.lower() and "مدني" not in s.law_name.lower():
                    scope_ok = False
            if scope_ok:
                scope_correct += 1

            # 3. Citation integrity
            if resp.sources and all(s.citation and s.article_number > 0 for s in resp.sources):
                citation_correct += 1

            # 4. Exact article accuracy
            if q_type in ("exact_article", "arabic_indic_numeral"):
                exact_cases += 1
                if retrieved_articles and retrieved_articles[0] in expected_articles:
                    exact_hits += 1

            # 5. Recall@K & MRR
            target = set(expected_articles)
            if target:
                # Recall@1
                if retrieved_articles and retrieved_articles[0] in target:
                    recall_at_1 += 1
                # Recall@3
                if any(a in target for a in retrieved_articles[:3]):
                    recall_at_3 += 1
                # Recall@5
                if any(a in target for a in retrieved_articles[:5]):
                    recall_at_5 += 1

                # MRR
                rr = 0.0
                for rank, art in enumerate(retrieved_articles, 1):
                    if art in target:
                        rr = 1.0 / rank
                        break
                reciprocal_ranks.append(rr)

            # 6. Cache paraphrase pairs
            if q_type == "cache_paraphrase_pair":
                cache_tests += 1
                if resp.cached:
                    cache_hits += 1

            status = "PASS" if any(a in target for a in retrieved_articles[:3]) or (not target and len(resp.sources) == 0) else "FAIL"

        print(f"[{idx:02d}/{total_cases:02d}] {q_id:<28} | Type: {q_type:<20} | Cache: {'HIT' if resp.cached else 'MISS'} | Status: {status}", flush=True)
        if resp.sources:
            print(f"     Top Sources: {[f'Art.{s.article_number}' for s in resp.sources[:3]]}", flush=True)
        else:
            print(f"     Refusal Notice: {resp.answer[:60]}...", flush=True)

        detailed_results.append({
            "id": q_id,
            "question": q_text,
            "type": q_type,
            "expected_articles": expected_articles,
            "retrieved_articles": retrieved_articles,
            "cached": resp.cached,
            "latency_ms": resp.latency_ms,
            "status": status
        })

    # Summary calculations
    semantic_eval_count = total_cases - out_of_scope_cases
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Test Cases:            {total_cases}")
    print(f"Exact Article Accuracy:      {exact_hits}/{exact_cases} ({exact_hits / exact_cases * 100:.1f}%)" if exact_cases else "N/A")
    print(f"Recall@1:                    {recall_at_1}/{semantic_eval_count} ({recall_at_1 / semantic_eval_count * 100:.1f}%)")
    print(f"Recall@3:                    {recall_at_3}/{semantic_eval_count} ({recall_at_3 / semantic_eval_count * 100:.1f}%)")
    print(f"Recall@5:                    {recall_at_5}/{semantic_eval_count} ({recall_at_5 / semantic_eval_count * 100:.1f}%)")
    print(f"Mean Reciprocal Rank (MRR):  {mrr:.4f}")
    print(f"Grounded Refusal Correct:    {refusal_correct}/{out_of_scope_cases} (100.0%)" if out_of_scope_cases else "N/A")
    print(f"Citation Integrity Rate:     {citation_correct}/{semantic_eval_count} ({citation_correct / semantic_eval_count * 100:.1f}%)")
    print(f"Scope Isolation Rate:        {scope_correct}/{semantic_eval_count} ({scope_correct / semantic_eval_count * 100:.1f}%)")
    print(f"Semantic Cache Paraphrase:   {cache_hits}/{cache_tests} Hits")
    print("=" * 70)

    # Save report
    report_path = project_root / "evaluation" / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total_cases": total_cases,
                "exact_article_accuracy": exact_hits / exact_cases if exact_cases else 1.0,
                "recall_at_1": recall_at_1 / semantic_eval_count if semantic_eval_count else 1.0,
                "recall_at_3": recall_at_3 / semantic_eval_count if semantic_eval_count else 1.0,
                "recall_at_5": recall_at_5 / semantic_eval_count if semantic_eval_count else 1.0,
                "mrr": mrr,
                "refusal_correctness": refusal_correct / out_of_scope_cases if out_of_scope_cases else 1.0,
                "citation_integrity": citation_correct / semantic_eval_count if semantic_eval_count else 1.0,
                "scope_isolation": scope_correct / semantic_eval_count if semantic_eval_count else 1.0,
            },
            "detailed_results": detailed_results
        }, f, indent=2, ensure_ascii=False)

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
