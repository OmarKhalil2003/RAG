import pytest
from legal_rag.models import RAGResponse, SourceCitation, QuerySignals, UserRole
from legal_rag.evaluation.ragas_adapter import (
    prepare_sample,
    build_ragas_dataset,
    DeterministicLegalEvaluator,
    execute_ragas_evaluation
)


def test_prepare_sample_structure():
    response = RAGResponse(
        answer="وفقاً للمادة 147، العقد شريعة المتعاقدين.",
        sources=[
            SourceCitation(
                document_id="egyptian_civil_code_1948",
                law_name="القانون المدني المصري",
                article_number=147,
                citation="Article 147",
                text_ar="العقد شريعة المتعاقدين",
                text_en="The contract makes the law of the parties",
                is_repealed=False
            )
        ],
        cached=False,
        retrieval_count=1,
        query_signals=QuerySignals(original_query="ما هي المادة 147؟", normalized_query="ما هي المادة 147؟", language="ar"),
        latency_ms=45.2
    )

    sample = prepare_sample(
        question="ما هي المادة 147؟",
        response=response,
        ground_truth="العقد شريعة المتعاقدين ولا يجوز نقضه إلا باتفاق الطرفين.",
        metadata={"id": "test_1", "expected_articles": [147], "expected_law_type": "civil"}
    )

    assert sample["question"] == "ما هي المادة 147؟"
    assert "147" in sample["answer"]
    assert len(sample["contexts"]) == 1
    assert "العقد شريعة المتعاقدين" in sample["contexts"][0]
    assert sample["retrieved_articles"] == [147]
    assert sample["ground_truth"].startswith("العقد شريعة")


def test_build_ragas_dataset():
    samples = [
        {
            "question": "Q1",
            "answer": "A1",
            "contexts": ["C1"],
            "ground_truth": "G1",
            "retrieved_articles": [1],
            "cached": False,
            "latency_ms": 10.0,
            "metadata": {}
        },
        {
            "question": "Q2",
            "answer": "A2",
            "contexts": ["C2"],
            "ground_truth": "G2",
            "retrieved_articles": [2],
            "cached": False,
            "latency_ms": 12.0,
            "metadata": {}
        }
    ]
    ds = build_ragas_dataset(samples)
    assert len(ds) == 2
    assert "question" in ds.column_names
    assert "answer" in ds.column_names
    assert "contexts" in ds.column_names
    assert "ground_truth" in ds.column_names


def test_deterministic_evaluator_metrics():
    evaluator = DeterministicLegalEvaluator()
    scores = evaluator.evaluate_sample(
        question="ما هي أحكام المادة 157 من القانون المدني؟",
        answer="تنص المادة 157 على أنه في العقود الملزمة للجانبين إذا لم يوف أحد المتعاقدين بالتزامه جاز للمتعاقد الآخر المطالبة بالفسخ أو التنفيذ مع التعويض.",
        contexts=["في العقود الملزمة للجانبين، إذا لم يوف أحد المتعاقدين بالتزامه جاز للمتعاقد الآخر بعد إعذاره المدين أن يطالب بتنفيذ العقد أو بفسخه، مع التعويض في الحالتين إن كان له مقتض."],
        ground_truth="المادة 157 تنظم الفسخ والتعويض في العقود التبادلية.",
        expected_articles=[157],
        retrieved_articles=[157, 158]
    )

    assert "faithfulness" in scores
    assert "answer_relevancy" in scores
    assert "context_precision" in scores
    assert "context_recall" in scores
    assert scores["faithfulness"] > 0.7
    assert scores["context_precision"] == 1.0  # article 157 is rank 1
    assert scores["context_recall"] == 1.0     # article 157 is found


def test_execute_ragas_evaluation_mock_mode():
    samples = [
        {
            "question": "ما هو نطاق سريان قانون التحكيم؟",
            "answer": "يسري على كل تحكيم مدني أو تجاري يجري في مصر وفقاً للمادة 1.",
            "contexts": ["تسري أحكام هذا القانون على كل تحكيم بين أطراف من أشخاص القانون العام أو الخاص إذا كان التحكيم يجري في مصر."],
            "ground_truth": "يسري قانون التحكيم على التحكيم الداخلي والدولي في مصر.",
            "retrieved_articles": [1],
            "cached": False,
            "latency_ms": 15.0,
            "metadata": {"id": "arb_1", "expected_law_type": "arbitration", "expected_articles": [1]}
        }
    ]

    result = execute_ragas_evaluation(samples, provider="mock")
    assert result["total_cases"] == 1
    assert "aggregate_scores" in result
    agg = result["aggregate_scores"]
    assert 0.0 <= agg["faithfulness"] <= 1.0
    assert 0.0 <= agg["context_precision"] <= 1.0
    assert 0.0 <= agg["context_recall"] <= 1.0
    assert len(result["per_case_results"]) == 1
