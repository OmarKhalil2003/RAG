import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_rag.pipeline import RAGService
from legal_rag.models import UserRole
from legal_rag.evaluation.ragas_adapter import prepare_sample, execute_ragas_evaluation


def run_ragas_benchmark(
    statute: str = "all",
    sample_limit: int | None = None,
    provider: str = "auto",
    output_path: Path | None = None
):
    print("=" * 75)
    print("      JURIS-EGYPT: RAGAS COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 75)

    dataset_path = project_root / "evaluation" / "questions.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    # Filter by statute
    if statute.lower() != "all":
        cases = [c for c in all_cases if c.get("expected_law_type", "civil").lower() == statute.lower()]
    else:
        cases = all_cases

    if sample_limit and sample_limit > 0:
        cases = cases[:sample_limit]

    print(f"Loaded {len(cases)} evaluation test cases (Statute Filter: {statute.upper()}, Provider: {provider.upper()})\n")

    rag = RAGService()
    samples = []

    print("Running legal inquiries through RAG pipeline...")
    start_time = time.perf_counter()

    for idx, case in enumerate(cases, 1):
        q_text = case["question"]
        expected_law = case.get("expected_law_type", "civil")
        expected_jurisdiction = case.get("expected_jurisdiction", "Egypt")
        ground_truth = case.get("ground_truth", "")

        t0 = time.perf_counter()
        resp = rag.ask(
            question=q_text,
            role=UserRole.LAWYER,
            jurisdiction=expected_jurisdiction,
            law_type=expected_law
        )
        latency = (time.perf_counter() - t0) * 1000.0

        sample = prepare_sample(
            question=q_text,
            response=resp,
            ground_truth=ground_truth,
            metadata={
                "id": case.get("id", f"case_{idx}"),
                "expected_law_type": expected_law,
                "expected_articles": case.get("expected_articles", []),
                "type": case.get("type", "standard"),
                "language": case.get("language", "ar"),
                "latency_ms": latency
            }
        )
        samples.append(sample)
        print(f"  [{idx:02d}/{len(cases):02d}] {case.get('id')}: {len(resp.sources)} sources ({latency:.1f} ms)")

    total_pipeline_sec = time.perf_counter() - start_time
    print(f"\nPipeline execution finished in {total_pipeline_sec:.2f} seconds.")
    print("Evaluating RAGAS metrics (Faithfulness, Relevance, Precision, Recall)...")

    eval_results = execute_ragas_evaluation(samples, provider=provider)

    # Separate per-statute breakdown
    per_statute = {}
    for case in eval_results["per_case_results"]:
        law = case.get("law_type", "civil")
        if law not in per_statute:
            per_statute[law] = []
        per_statute[law].append(case["scores"])

    statute_breakdown = {}
    for law, score_list in per_statute.items():
        statute_breakdown[law] = {
            "count": len(score_list),
            "faithfulness": round(float(np.mean([s["faithfulness"] for s in score_list])), 4),
            "answer_relevancy": round(float(np.mean([s["answer_relevancy"] for s in score_list])), 4),
            "context_precision": round(float(np.mean([s["context_precision"] for s in score_list])), 4),
            "context_recall": round(float(np.mean([s["context_recall"] for s in score_list])), 4)
        }

    agg = eval_results["aggregate_scores"]

    print("\n" + "=" * 75)
    print("                       RAGAS BENCHMARK SCORECARD")
    print("=" * 75)
    print(f"Evaluator Mode:       {eval_results['provider'].upper()}")
    print(f"Total Cases Tested:   {eval_results['total_cases']}")
    print("-" * 75)
    print(f"Faithfulness Score:   {agg['faithfulness'] * 100:.1f}%  (Non-hallucination & grounding)")
    print(f"Answer Relevancy:     {agg['answer_relevancy'] * 100:.1f}%  (Pertinence to legal inquiry)")
    print(f"Context Precision:    {agg['context_precision'] * 100:.1f}%  (Signal-to-noise ratio in ranking)")
    print(f"Context Recall:       {agg['context_recall'] * 100:.1f}%  (Statutory article coverage)")
    print("=" * 75)

    print("\nStatutory Breakdown:")
    for law, metrics in statute_breakdown.items():
        law_title = "Egyptian Civil Code" if law == "civil" else "Egyptian Arbitration Law"
        print(f"  • {law_title} ({metrics['count']} cases):")
        print(f"      Faithfulness: {metrics['faithfulness'] * 100:.1f}% | Relevancy: {metrics['answer_relevancy'] * 100:.1f}% | Precision: {metrics['context_precision'] * 100:.1f}% | Recall: {metrics['context_recall'] * 100:.1f}%")

    # Serialize report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": eval_results["provider"],
        "total_cases": eval_results["total_cases"],
        "aggregate_scores": agg,
        "statute_breakdown": statute_breakdown,
        "per_case_results": eval_results["per_case_results"]
    }

    out_file = output_path or (project_root / "evaluation" / "ragas_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Also emit clean Markdown summary
    summary_md_path = project_root / "evaluation" / "ragas_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# RAGAS Benchmark Summary Report\n\n")
        f.write(f"- **Evaluator Mode:** `{eval_results['provider']}`\n")
        f.write(f"- **Total Cases:** `{eval_results['total_cases']}`\n\n")
        f.write("| RAGAS Metric | Score | Description |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Faithfulness** | **{agg['faithfulness'] * 100:.1f}%** | Grounding verification; zero hallucination tolerance |\n")
        f.write(f"| **Answer Relevancy** | **{agg['answer_relevancy'] * 100:.1f}%** | Direct pertinence to the statutory question asked |\n")
        f.write(f"| **Context Precision** | **{agg['context_precision'] * 100:.1f}%** | Authoritative provisions prioritized in candidate ranking |\n")
        f.write(f"| **Context Recall** | **{agg['context_recall'] * 100:.1f}%** | Complete coverage of necessary statutory provisions |\n\n")
        f.write("## Per-Statute Breakdown\n\n")
        for law, metrics in statute_breakdown.items():
            law_title = "Egyptian Civil Code (Law 131/1948)" if law == "civil" else "Egyptian Arbitration Law (Law 27/1994)"
            f.write(f"### {law_title}\n")
            f.write(f"- Cases: {metrics['count']}\n")
            f.write(f"- Faithfulness: `{metrics['faithfulness'] * 100:.1f}%`\n")
            f.write(f"- Answer Relevancy: `{metrics['answer_relevancy'] * 100:.1f}%`\n")
            f.write(f"- Context Precision: `{metrics['context_precision'] * 100:.1f}%`\n")
            f.write(f"- Context Recall: `{metrics['context_recall'] * 100:.1f}%`\n\n")

    print(f"\nSerialized full JSON report to: {out_file}")
    print(f"Serialized Markdown summary to: {summary_md_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Legal RAG pipeline.")
    parser.add_argument("--statute", choices=["all", "civil", "arbitration"], default="all", help="Statute to evaluate")
    parser.add_argument("--sample", type=int, default=None, help="Sample limit for fast evaluation")
    parser.add_argument("--provider", choices=["auto", "gemini", "openai", "mock"], default="auto", help="LLM judge provider")
    parser.add_argument("--offline", action="store_true", help="Run deterministic offline evaluation without external LLM calls")
    args = parser.parse_args()

    selected_provider = "mock" if args.offline else args.provider

    run_ragas_benchmark(
        statute=args.statute,
        sample_limit=args.sample,
        provider=selected_provider
    )
